# requirements.txt - Updated compatible versions
"""
langgraph>=0.2.0
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-community>=0.2.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
googlesearch-python>=1.2.3
duckduckgo-search>=6.1.0
aiohttp>=3.9.1
beautifulsoup4>=4.12.2
redis>=5.0.1
numpy>=1.24.3
scikit-learn>=1.3.0
openai>=1.0.0
"""
# 293

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, TypedDict, Annotated, Any, Tuple
from enum import Enum
import hashlib
import logging
import re
import urllib.parse

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity

from langchain_openai import AzureChatOpenAI
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langchain.tools import Tool

import aiohttp
from bs4 import BeautifulSoup
import redis
from googlesearch import search as google_search
from duckduckgo_search import DDGS # Corrected import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models
llm  = AzureChatOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=Azure_endpoint,
    api_version="2024-12-01-preview",
    model="gpt-4o-mini",
    streaming=True,
    temperature=0.8,
    max_tokens=512,
    azure_deployment="gpt-4o-mini",  # Ensure this matches your deployment name
)
embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Redis for caching (optional)
try:
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
except:
    redis_client = None
    logger.warning("Redis not available, caching disabled")

# Constants
CACHE_TTL = 3600  # 1 hour
MAX_RESULTS_PER_SOURCE = 10
RATE_LIMIT_DELAY = 0.5

class QueryIntent(Enum):
    FACTUAL = "factual"
    NAVIGATIONAL = "navigational"
    INFORMATIONAL = "informational"
    TRANSACTIONAL = "transactional"
    RESEARCH = "research"

class SearchResult(TypedDict):
    title: str
    url: str
    snippet: str
    source: str
    timestamp: str
    relevance_score: float
    authority_score: float
    freshness_score: float
    verified: bool
    content: Optional[str]

class AgentState(TypedDict):
    query: str
    intent: Optional[QueryIntent]
    expanded_queries: List[str]
    search_results: List[SearchResult]
    semantic_index: Optional[Any]  # FAISS index
    ranked_results: List[SearchResult]
    verified_facts: List[Dict[str, Any]]
    answer: str
    confidence_score: float
    error_log: List[str]
    cache_hits: int
    processing_time: float
    user_context: Dict[str, Any]
    iteration: int

class SearchAgent:
    def __init__(self):
        self.memory = {}
        self.user_profiles = {}

    async def classify_intent(self, state: AgentState) -> AgentState:
        """Classify the query intent to optimize search strategy"""
        try:
            prompt = f"""
            Classify the following search query into one of these intents:
            - FACTUAL: Looking for specific facts or data
            - NAVIGATIONAL: Looking for a specific website or resource
            - INFORMATIONAL: Seeking general information about a topic
            - TRANSACTIONAL: Looking to perform an action or transaction
            - RESEARCH: In-depth research requiring multiple sources

            Query: {state['query']}

            Return only the intent category.
            """

            response = await llm.ainvoke(prompt)
            intent_str = response.content.strip().upper()
            state['intent'] = QueryIntent[intent_str]

        except Exception as e:
            state['error_log'].append(f"Intent classification error: {str(e)}")
            state['intent'] = QueryIntent.INFORMATIONAL

        return state

    async def expand_query(self, state: AgentState) -> AgentState:
        """Expand and refine the query for better results"""
        try:
            prompt = f"""
            Given the search query and intent, generate 3-5 expanded or related queries
            that would help find comprehensive information.

            Original Query: {state['query']}
            Intent: {state['intent'].value if state['intent'] else 'unknown'}

            Return queries as a JSON list.
            """

            response = await llm.ainvoke(prompt)
            expanded = json.loads(response.content)
            state['expanded_queries'] = [state['query']] + expanded[:4]

        except Exception as e:
            state['error_log'].append(f"Query expansion error: {str(e)}")
            state['expanded_queries'] = [state['query']]

        return state

    async def _fetch_snippet(self, url: str) -> str:
        """Fetch snippet from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        # Extract meta description or first paragraph
                        meta_desc = soup.find('meta', attrs={'name': 'description'})
                        if meta_desc:
                            return meta_desc.get('content', '')[:300]
                        # Fallback to first paragraph
                        p = soup.find('p')
                        if p:
                            return p.get_text()[:300]
        except Exception as e:
            logger.error(f"Error fetching snippet from {url}: {str(e)}")
        return ""

    async def _fetch_content(self, url: str) -> str:
        """Fetch full content from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        # Get text content
                        text = soup.get_text()
                        # Clean up whitespace
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        text = '\n'.join(chunk for chunk in chunks if chunk)
                        return text[:5000]  # Limit content length
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {str(e)}")
        return ""

    def _calculate_authority(self, url: str) -> float:
        """Calculate authority score based on domain"""
        try:
            domain = urllib.parse.urlparse(url).netloc.lower()

            # High authority domains
            high_authority = ['wikipedia.org', 'gov', 'edu', 'nature.com', 'ieee.org']
            medium_authority = ['medium.com', 'reddit.com', 'stackoverflow.com']

            if any(auth in domain for auth in high_authority):
                return 0.9
            elif any(auth in domain for auth in medium_authority):
                return 0.6
            elif domain.endswith('.org'):
                return 0.7
            elif domain.endswith('.com'):
                return 0.5
            else:
                return 0.3

        except Exception:
            return 0.3

    def _calculate_freshness(self, timestamp: str) -> float:
        """Calculate freshness score based on timestamp"""
        try:
            time_diff = datetime.now() - datetime.fromisoformat(timestamp)
            days_old = time_diff.days

            if days_old <= 1:
                return 1.0
            elif days_old <= 7:
                return 0.8
            elif days_old <= 30:
                return 0.6
            elif days_old <= 90:
                return 0.4
            else:
                return 0.2

        except Exception:
            return 0.5

    async def search_google(self, query: str) -> List[SearchResult]:
        """Search using Google"""
        results = []
        try:
            # Check cache first
            cache_key = f"google:{hashlib.md5(query.encode()).hexdigest()}"
            if redis_client:
                cached = redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)

            # Rate limiting
            await asyncio.sleep(RATE_LIMIT_DELAY)

            for i, url in enumerate(google_search(query, num_results=MAX_RESULTS_PER_SOURCE)):
                if i >= MAX_RESULTS_PER_SOURCE:
                    break

                # Fetch snippet
                snippet = await self._fetch_snippet(url)

                result = SearchResult(
                    title=url.split('/')[2] if len(url.split('/')) > 2 else url,
                    url=url,
                    snippet=snippet,
                    source="google",
                    timestamp=datetime.now().isoformat(),
                    relevance_score=0.0,
                    authority_score=0.0,
                    freshness_score=0.0,
                    verified=False,
                    content=None
                )
                results.append(result)

            # Cache results
            if redis_client and results:
                redis_client.setex(cache_key, CACHE_TTL, json.dumps(results))

        except Exception as e:
            logger.error(f"Google search error: {str(e)}")

        return results

    async def search_duckduckgo(self, query: str) -> List[SearchResult]:
        """Search using DuckDuckGo"""
        results = []
        try:
            # Check cache
            cache_key = f"ddg:{hashlib.md5(query.encode()).hexdigest()}"
            if redis_client:
                cached = redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)

            async with DDGS() as ddgs: # Use DDGS directly in async with
                search_results = await ddgs.text(query, max_results=MAX_RESULTS_PER_SOURCE)

                for r in search_results:
                    result = SearchResult(
                        title=r.get('title', ''),
                        url=r.get('href', ''),
                        snippet=r.get('body', ''),
                        source="duckduckgo",
                        timestamp=datetime.now().isoformat(),
                        relevance_score=0.0,
                        authority_score=0.0,
                        freshness_score=0.0,
                        verified=False,
                        content=None
                    )
                    results.append(result)

            # Cache results
            if redis_client and results:
                redis_client.setex(cache_key, CACHE_TTL, json.dumps(results))

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {str(e)}")

        return results

    async def parallel_search(self, state: AgentState) -> AgentState:
        """Execute parallel searches across multiple sources"""
        all_results = []

        for query in state['expanded_queries']:
            # Create search tasks
            tasks = [
                self.search_google(query),
                self.search_duckduckgo(query),
            ]

            # Execute in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine results
            for result_set in results:
                if isinstance(result_set, list):
                    all_results.extend(result_set)

        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result['url'] not in seen_urls:
                seen_urls.add(result['url'])
                unique_results.append(result)

        state['search_results'] = unique_results
        return state

    def create_semantic_index(self, state: AgentState) -> AgentState:
        """Create FAISS index for semantic search"""
        try:
            if not state['search_results']:
                return state

            # Extract text for embedding
            texts = [f"{r['title']} {r['snippet']}" for r in state['search_results']]

            # Generate embeddings
            embeddings = embeddings_model.encode(texts)

            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings).astype('float32'))

            state['semantic_index'] = {
                'index': index,
                'embeddings': embeddings,
                'texts': texts
            }

        except Exception as e:
            state['error_log'].append(f"Semantic index creation error: {str(e)}")

        return state

    def calculate_scores(self, state: AgentState) -> AgentState:
        """Calculate relevance, authority, and freshness scores"""
        try:
            query_embedding = embeddings_model.encode([state['query']])[0]

            for i, result in enumerate(state['search_results']):
                # Relevance score (semantic similarity)
                if state.get('semantic_index') and i < len(state['semantic_index']['embeddings']):
                    result_embedding = state['semantic_index']['embeddings'][i]
                    relevance = cosine_similarity(
                        [query_embedding],
                        [result_embedding]
                    )[0][0]
                    result['relevance_score'] = float(relevance)

                # Authority score (based on domain and source)
                authority = self._calculate_authority(result['url'])
                result['authority_score'] = authority

                # Freshness score
                freshness = self._calculate_freshness(result['timestamp'])
                result['freshness_score'] = freshness

        except Exception as e:
            state['error_log'].append(f"Score calculation error: {str(e)}")

        return state

    def rank_results(self, state: AgentState) -> AgentState:
        """Rank results using multiple factors"""
        try:
            # Calculate composite scores
            for result in state['search_results']:
                result['composite_score'] = (
                    0.5 * result.get('relevance_score', 0) +
                    0.3 * result.get('authority_score', 0) +
                    0.2 * result.get('freshness_score', 0)
                )

            # Sort by composite score
            state['ranked_results'] = sorted(
                state['search_results'],
                key=lambda x: x.get('composite_score', 0),
                reverse=True
            )[:20]  # Top 20 results

        except Exception as e:
            state['error_log'].append(f"Ranking error: {str(e)}")
            state['ranked_results'] = state['search_results'][:20]

        return state

    async def _extract_and_verify_facts(self, content: str, query: str) -> List[Dict[str, Any]]:
        """Extract and verify facts from content"""
        try:
            prompt = f"""
            Extract key facts from the following content that are relevant to the query: "{query}"

            Content: {content[:2000]}

            Return a JSON list of facts with their confidence scores (0-1).
            Format: [{{"fact": "fact statement", "confidence": 0.95}}]
            """

            response = await llm.ainvoke(prompt)
            facts = json.loads(response.content)
            return facts

        except Exception as e:
            logger.error(f"Fact extraction error: {str(e)}")
            return []

    async def verify_facts(self, state: AgentState) -> AgentState:
        """Verify facts from top results"""
        try:
            # Extract potential facts from top results
            top_results = state['ranked_results'][:5]

            facts = []
            for result in top_results:
                # Fetch full content if needed
                if not result.get('content'):
                    result['content'] = await self._fetch_content(result['url'])

                # Extract and verify facts
                if result['content']:
                    verified_facts = await self._extract_and_verify_facts(
                        result['content'],
                        state['query']
                    )
                    facts.extend(verified_facts)

            state['verified_facts'] = facts

        except Exception as e:
            state['error_log'].append(f"Fact verification error: {str(e)}")
            state['verified_facts'] = []

        return state

    async def generate_answer(self, state: AgentState) -> AgentState:
        """Generate final answer with confidence score"""
        try:
            # Prepare context from top results
            context = "\n\n".join([
                f"Source: {r['url']}\nTitle: {r['title']}\nContent: {r['snippet']}"
                for r in state['ranked_results'][:5]
            ])

            # Include verified facts
            facts_context = "\n".join([
                f"Verified Fact: {f['fact']} (Confidence: {f['confidence']})"
                for f in state.get('verified_facts', [])
            ])

            prompt = f"""
            Based on the search results and verified facts, provide a comprehensive answer to the query.
            Include source citations and indicate confidence level.

            Query: {state['query']}
            Intent: {state['intent'].value if state['intent'] else 'unknown'}

            Search Results:
            {context}

            Verified Facts:
            {facts_context}

            Provide a detailed answer with source citations. End with a confidence score (0-1).
            """

            response = await llm.ainvoke(prompt)
            answer = response.content

            # Extract confidence score from answer
            confidence_match = re.search(r'confidence.*?(\d+\.?\d*)', answer.lower())
            if confidence_match:
                state['confidence_score'] = float(confidence_match.group(1))
            else:
                state['confidence_score'] = 0.7  # Default confidence

            state['answer'] = answer

        except Exception as e:
            state['error_log'].append(f"Answer generation error: {str(e)}")
            state['answer'] = "I apologize, but I encountered an error while generating the answer."
            state['confidence_score'] = 0.0

        return state

    def should_expand_query(self, state: AgentState) -> str:
        """Determine if query should be expanded"""
        if len(state['query'].split()) <= 2:
            return "expand"
        return "search"

    def should_create_index(self, state: AgentState) -> str:
        """Determine if semantic index should be created"""
        if len(state['search_results']) > 5:
            return "create_index"
        return "calculate_scores"

    def should_verify_facts(self, state: AgentState) -> str:
        """Determine if facts should be verified"""
        if state['intent'] == QueryIntent.FACTUAL and len(state['ranked_results']) > 0:
            return "verify"
        return "generate_answer"

def create_search_workflow() -> StateGraph:
    """Create the search workflow graph"""
    agent = SearchAgent()

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classify_intent", agent.classify_intent)
    workflow.add_node("expand_query", agent.expand_query)
    workflow.add_node("parallel_search", agent.parallel_search)
    workflow.add_node("create_semantic_index", agent.create_semantic_index)
    workflow.add_node("calculate_scores", agent.calculate_scores)
    workflow.add_node("rank_results", agent.rank_results)
    workflow.add_node("verify_facts", agent.verify_facts)
    workflow.add_node("generate_answer", agent.generate_answer)

    # Set entry point
    workflow.set_entry_point("classify_intent")

    # Add edges
    workflow.add_edge("classify_intent", "expand_query")
    workflow.add_edge("expand_query", "parallel_search")

    # Conditional routing
    workflow.add_conditional_edges(
        "parallel_search",
        agent.should_create_index,
        {
            "create_index": "create_semantic_index",
            "calculate_scores": "calculate_scores"
        }
    )

    workflow.add_edge("create_semantic_index", "calculate_scores")
    workflow.add_edge("calculate_scores", "rank_results")

    workflow.add_conditional_edges(
        "rank_results",
        agent.should_verify_facts,
        {
            "verify": "verify_facts",
            "generate_answer": "generate_answer"
        }
    )

    workflow.add_edge("verify_facts", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()

# Usage example
async def main():
    """Main function to demonstrate the search agent"""
    # Create the workflow
    app = create_search_workflow()

    # Initialize state
    initial_state = AgentState(
        query="What are the latest developments in quantum computing?",
        intent=None,
        expanded_queries=[],
        search_results=[],
        semantic_index=None,
        ranked_results=[],
        verified_facts=[],
        answer="",
        confidence_score=0.0,
        error_log=[],
        cache_hits=0,
        processing_time=0.0,
        user_context={},
        iteration=0
    )

    # Run the workflow
    start_time = time.time()

    try:
        final_state = await app.ainvoke(initial_state)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Print results
        print(f"Query: {final_state['query']}")
        print(f"Intent: {final_state['intent']}")
        print(f"Expanded Queries: {final_state['expanded_queries']}")
        print(f"Total Results: {len(final_state['search_results'])}")
        print(f"Top Results: {len(final_state['ranked_results'])}")
        print(f"Verified Facts: {len(final_state['verified_facts'])}")
        print(f"Confidence Score: {final_state['confidence_score']}")
        print(f"Processing Time: {processing_time:.2f}s")

        if final_state['error_log']:
            print(f"Errors: {final_state['error_log']}")

        print(f"\nAnswer:\n{final_state['answer']}")

    except Exception as e:
        print(f"Error running workflow: {str(e)}")

# Directly await main() instead of using asyncio.run()
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main()) # Keep asyncio.run() but apply nest_asyncio for Colab compatibility