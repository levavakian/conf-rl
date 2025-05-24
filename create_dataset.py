#!/usr/bin/env python3
"""
Script to collect resolved prediction markets data from Manifold Markets and Polymarket APIs.

This script fetches all resolved markets between two dates from both platforms and outputs
a dataset with standardized format including source, market description, and resolution.
"""

import asyncio
import aiohttp
import csv
import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
START_DATE = "2024-08-01"  # YYYY-MM-DD format
END_DATE = "2025-05-24"    # YYYY-MM-DD format
MANIFOLD_MIN_TRADERS = 30   # Minimum number of unique traders for Manifold markets
POLYMARKET_MIN_VOLUME = 100.0  # Minimum volume for Polymarket markets
OUTPUT_FORMAT = "json"      # Options: "csv", "json", "both"
OUTPUT_FILE = "resolved_markets"  # Output filename prefix

# Manifold Markets API Configuration
MANIFOLD_API_BASE = "https://api.manifold.markets"

# Convert START_DATE and END_DATE to timestamps (Unix milliseconds)
START_TIMESTAMP = int(datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
END_TIMESTAMP = int(datetime.strptime(END_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

async def search_manifold_markets(
    session: aiohttp.ClientSession,
    term: str = "",
    sort: str = "resolve-date", 
    filter_state: str = "resolved",
    contract_type: str = "BINARY",
    topic_slug: Optional[str] = None,
    creator_id: Optional[str] = None,
    limit: int = 500,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Call the Manifold Markets /v0/search-markets endpoint.
    
    Args:
        session: aiohttp ClientSession for making requests
        term: Search query (can be empty string)
        sort: Sort order - one of: most-popular (default), newest, score, daily-score, 
              freshness-score, 24-hour-vol, liquidity, subsidy, last-updated, close-date,
              start-time, resolve-date, random, bounty-amount, prob-descending, prob-ascending
        filter_state: Closing state - one of: all (default), open, closed, resolved, news,
                     closing-90-days, closing-week, closing-month, closing-day
        contract_type: Type - ALL (default), BINARY (yes/no), MULTIPLE_CHOICE, BOUNTY, POLL
        topic_slug: Only include questions with this topic tag slug
        creator_id: Only include questions created by this user ID
        limit: Number of contracts to return (0-1000, default 100)
        offset: Number of contracts to skip for pagination
        
    Returns:
        Dict containing the API response with list of markets
        
    Raises:
        aiohttp.ClientError: If the API request fails
        ValueError: If response is not valid JSON
    """
    url = f"{MANIFOLD_API_BASE}/v0/search-markets"
    
    params = {
        "term": term,
        "sort": sort,
        "filter": filter_state,
        "contractType": contract_type,
        "limit": limit,
        "offset": offset
    }
    
    # Add optional parameters if provided
    if topic_slug:
        params["topicSlug"] = topic_slug
    if creator_id:
        params["creatorId"] = creator_id
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            logger.info(f"Retrieved {len(data)} markets from Manifold API")
            return data
            
    except aiohttp.ClientError as e:
        logger.error(f"HTTP error calling Manifold search API: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from Manifold API: {e}")
        raise ValueError(f"Invalid JSON response: {e}")


async def ManifoldGenerator():
    """Generator function to fetch markets from Manifold Markets API."""
    async with aiohttp.ClientSession() as session:
        n = 0
        live = True

        while live:
            markets = await search_manifold_markets(
                session=session,
                term="",  # Empty search term to get all markets
                sort="newest",
                filter_state="resolved",
                contract_type="BINARY",
                limit=500,
                offset=n,
            )

            if not markets:
                break
            
            n += len(markets)
            
            if markets:
                last_market = markets[-1]
                last_resolution_time = last_market.get('resolutionTime', 0)
                if last_resolution_time < START_TIMESTAMP:
                    live = False

            # Filter markets based on resolutionTime and minimum traders using inline predicate
            is_valid_market = lambda market: (
                START_TIMESTAMP <= market.get('resolutionTime', 0) <= END_TIMESTAMP and
                market.get('uniqueBettorCount', 0) >= MANIFOLD_MIN_TRADERS
            )

            convert_manifold_market = lambda market: {
                "source": "manifold",
                "url": market.get('url', ''),
                "question": market.get('question', ''),
                "resolution": market.get('resolution', ''),
                "resolutionTime": market.get('resolutionTime', 0)
            }
            
            output = [convert_manifold_market(market) for market in markets if is_valid_market(market)]
            yield output

async def run_manifold():
    manifold_generator = ManifoldGenerator()
    manifold_data = []
    async for markets in manifold_generator:
        manifold_data.extend(markets)

    with open(f"manifold_markets_{START_DATE}_{END_DATE}.json", "w") as f:
        json.dump(manifold_data, f, indent=2)


async def main():
    await run_manifold()

if __name__ == "__main__":
    asyncio.run(main())