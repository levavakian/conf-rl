#!/usr/bin/env python3
"""
Script to collect resolved prediction markets data from Manifold Markets, Polymarket, and Metaculus APIs.

This script fetches all resolved markets between two dates from all three platforms and outputs
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
METACULUS_MIN_FORECASTERS = 10  # Minimum number of forecasters for Metaculus questions
OUTPUT_FORMAT = "json"      # Options: "csv", "json", "both"
OUTPUT_FILE = "resolved_markets"  # Output filename prefix

# API Configuration
MANIFOLD_API_BASE = "https://api.manifold.markets"
POLYMARKET_API_BASE = "https://gamma-api.polymarket.com"
METACULUS_API_BASE = "https://www.metaculus.com"

# Convert START_DATE and END_DATE to timestamps (Unix milliseconds)
START_TIMESTAMP = int(datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
END_TIMESTAMP = int(datetime.strptime(END_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

# Convert to ISO format for Polymarket (they use ISO date strings)
START_DATE_ISO = datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc).isoformat()
END_DATE_ISO = datetime.strptime(END_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc).isoformat()

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
        logger.debug(f"Making request to Manifold API: {url} with params: {params}")
        async with session.get(url, params=params) as response:
            logger.debug(f"Manifold API response status: {response.status}")
            response.raise_for_status()
            data = await response.json()
            logger.info(f"Retrieved {len(data)} markets from Manifold API")
            return data
            
    except aiohttp.ClientError as e:
        logger.error(f"HTTP error calling Manifold search API: {e}")
        logger.error(f"Request URL: {url}")
        logger.error(f"Request params: {params}")
        if hasattr(e, 'status'):
            logger.error(f"HTTP status code: {e.status}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from Manifold API: {e}")
        raise ValueError(f"Invalid JSON response: {e}")


async def fetch_metaculus_questions(
    session: aiohttp.ClientSession,
    offset: int = 0,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Fetch questions from Metaculus API.
    
    Args:
        session: aiohttp ClientSession for making requests
        page: Page number to fetch
        limit: Number of questions per page
        
    Returns:
        Dict containing the API response with list of questions
        
    Raises:
        aiohttp.ClientError: If the API request fails
        ValueError: If response is not valid JSON
    """
    url = f"{METACULUS_API_BASE}/api/posts/"
    
    params = {
        "offset": offset,
        "limit": limit,
        "statuses": ["resolved"],  # Only get resolved questions
        "forecast_type": ["binary"],     # Only get forecast questions (not discussions).
    }
    
    try:
        logger.debug(f"Making request to Metaculus API: {url} with params: {params}")
        async with session.get(url, params=params) as response:
            logger.debug(f"Metaculus API response status: {response.status}")
            
            # Log response headers for debugging
            logger.debug(f"Metaculus API response headers: {dict(response.headers)}")
            
            if response.status != 200:
                response_text = await response.text()
                logger.error(f"Metaculus API returned status {response.status}")
                logger.error(f"Response body: {response_text}")
                
            response.raise_for_status()
            data = await response.json()
            logger.info(f"Retrieved offset {offset} with {len(data.get('results', []))} questions from Metaculus API")
            logger.info(data.get('results', [])[0].get('title'))
            return data
            
    except aiohttp.ClientError as e:
        logger.error(f"HTTP error calling Metaculus API: {e}")
        logger.error(f"Request URL: {url}")
        logger.error(f"Request params: {params}")
        if hasattr(e, 'status'):
            logger.error(f"HTTP status code: {e.status}")
        
        # Try to get response body for more details
        try:
            if hasattr(e, 'response') and e.response:
                response_text = await e.response.text()
                logger.error(f"Error response body: {response_text}")
        except Exception as inner_e:
            logger.error(f"Could not read error response body: {inner_e}")
            
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from Metaculus API: {e}")
        raise ValueError(f"Invalid JSON response: {e}")


async def fetch_polymarket_markets(
    session: aiohttp.ClientSession,
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Fetch markets from Polymarket Gamma API.
    
    Args:
        session: aiohttp ClientSession for making requests
        limit: Number of markets per page
        offset: Pagination offset
        
    Returns:
        Dict containing the API response with list of markets
        
    Raises:
        aiohttp.ClientError: If the API request fails
        ValueError: If response is not valid JSON
    """
    url = f"{POLYMARKET_API_BASE}/markets"
    
    params = {
        "limit": limit,
        "offset": offset,
        "closed": "true",  # Only get closed/resolved markets
        "end_date_min": START_DATE_ISO,  # Filter by resolution date range
        "end_date_max": END_DATE_ISO,
        "volume_num_min": POLYMARKET_MIN_VOLUME,  # Minimum volume filter
        # Remove order parameter to avoid validation error
    }
    
    try:
        logger.debug(f"Making request to Polymarket API: {url} with params: {params}")
        async with session.get(url, params=params) as response:
            logger.debug(f"Polymarket API response status: {response.status}")
            
            if response.status != 200:
                response_text = await response.text()
                logger.error(f"Polymarket API returned status {response.status}")
                logger.error(f"Response body: {response_text}")
                
            response.raise_for_status()
            data = await response.json()
            logger.info(f"Retrieved {len(data)} markets from Polymarket API at offset {offset}")
            return data
            
    except aiohttp.ClientError as e:
        logger.error(f"HTTP error calling Polymarket API: {e}")
        logger.error(f"Request URL: {url}")
        logger.error(f"Request params: {params}")
        if hasattr(e, 'status'):
            logger.error(f"HTTP status code: {e.status}")
        
        # Try to get response body for more details
        try:
            if hasattr(e, 'response') and e.response:
                response_text = await e.response.text()
                logger.error(f"Error response body: {response_text}")
        except Exception as inner_e:
            logger.error(f"Could not read error response body: {inner_e}")
            
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from Polymarket API: {e}")
        raise ValueError(f"Invalid JSON response: {e}")


async def ManifoldGenerator():
    """Generator function to fetch markets from Manifold Markets API."""
    async with aiohttp.ClientSession() as session:
        n = 0
        live = True

        while live:
            try:
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
                
            except Exception as e:
                logger.error(f"Error in ManifoldGenerator: {e}")
                break


async def MetaculusGenerator():
    """Generator function to fetch questions from Metaculus API."""
    async with aiohttp.ClientSession() as session:
        offset = 0
        
        while True:
            try:
                data = await fetch_metaculus_questions(session, offset=offset, limit=500)
                questions = data.get('results', [])
                
                if not questions:
                    logger.info(f"No more questions found on offset {offset}, stopping")
                    break
                
                logger.info(f"Processing {len(questions)} questions from offset {offset}")
                
                # Filter questions based on resolution time and minimum forecasters
                def is_valid_question(question):
                    # Check if question has resolution time
                    resolve_time_str = question.get('question', {}).get('actual_resolve_time')
                    if not resolve_time_str:
                        return False
                    
                    try:
                        # Parse resolution time
                        resolve_time = datetime.fromisoformat(resolve_time_str.replace('Z', '+00:00'))
                        resolve_timestamp = int(resolve_time.timestamp() * 1000)
                        
                        # Check if within date range
                        if not (START_TIMESTAMP <= resolve_timestamp <= END_TIMESTAMP):
                            return False
                        
                        # Check if it has a resolution (0 or 1)
                        resolution = question.get('question', {}).get('resolution')
                        if resolution not in ["yes", "no"]:
                            return False
                        
                        forecasts_count = question.get('forecasts_count')
                        if forecasts_count < METACULUS_MIN_FORECASTERS:
                            return False
                        
                        return True
                        
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error parsing question data: {e}")
                        return False

                def convert_metaculus_question(question):
                    resolve_time_str = question.get('question', {}).get('actual_resolve_time', '')
                    resolve_timestamp = 0
                    
                    if resolve_time_str:
                        try:
                            resolve_time = datetime.fromisoformat(resolve_time_str.replace('Z', '+00:00'))
                            resolve_timestamp = int(resolve_time.timestamp() * 1000)
                        except ValueError:
                            pass
                    
                    return {
                        "source": "metaculus",
                        "url": f"https://www.metaculus.com/questions/{question.get('id', '')}/",
                        "question": question.get('question', {}).get('title', '') + "\n" + question.get('question', {}).get('resolution_criteria', ''),
                        "resolution": question.get('question', {}).get('resolution').upper(),
                        "resolutionTime": resolve_timestamp
                    }
                
                valid_questions = [convert_metaculus_question(q) for q in questions if is_valid_question(q)]
                logger.info(f"Found {len(valid_questions)} valid questions on offset {offset}")

                offset += len(questions)
                if valid_questions:
                    yield valid_questions
                
                # Check if we should continue (if we're getting questions outside our date range)
                if questions:
                    last_question = questions[-1]
                    last_resolve_time_str = last_question.get('question', {}).get('actual_resolve_time')
                    if last_resolve_time_str:
                        try:
                            last_resolve_time = datetime.fromisoformat(last_resolve_time_str.replace('Z', '+00:00'))
                            last_resolve_timestamp = int(last_resolve_time.timestamp() * 1000)
                            if last_resolve_timestamp < START_TIMESTAMP:
                                logger.info(f"Reached questions outside date range on offset {offset}, stopping")
                                break
                        except ValueError:
                            pass
                
                
            except Exception as e:
                logger.error(f"Error fetching Metaculus offset {offset}: {e}")
                break


async def PolymarketGenerator():
    """Generator function to fetch markets from Polymarket API."""
    async with aiohttp.ClientSession() as session:
        offset = 0
        
        while True:
            try:
                markets = await fetch_polymarket_markets(session, limit=500, offset=offset)
                
                if not markets:
                    logger.info(f"No more markets found at offset {offset}, stopping")
                    break
                
                logger.info(f"Processing {len(markets)} markets from offset {offset}")
                
                # Filter and convert markets
                def is_valid_market(market):
                    # Check if market has endDate (resolution date)
                    end_date_str = market.get('endDate')
                    if not end_date_str:
                        return False
                    
                    try:
                        # Parse end date
                        end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                        end_timestamp = int(end_date.timestamp() * 1000)
                        
                        # Check if within date range
                        if not (START_TIMESTAMP <= end_timestamp <= END_TIMESTAMP):
                            return False
                        
                        # Check if market is closed/resolved
                        if not market.get('closed', False):
                            return False
                        
                        # Check if UMA resolution is complete
                        if market.get('umaResolutionStatus') != 'resolved':
                            return False
                        
                        # Check volume requirement
                        volume = market.get('volumeNum', 0)
                        if volume < POLYMARKET_MIN_VOLUME:
                            return False
                        
                        # Check if we have outcome prices to determine resolution
                        outcome_prices = market.get('outcomePrices')
                        if not outcome_prices:
                            return False
                        
                        return True
                        
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error parsing market data: {e}")
                        return False

                def convert_polymarket_market(market):
                    end_date_str = market.get('endDate', '')
                    end_timestamp = 0
                    
                    if end_date_str:
                        try:
                            end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                            end_timestamp = int(end_date.timestamp() * 1000)
                        except ValueError:
                            pass
                    
                    # Determine resolution from outcome prices
                    resolution = "UNKNOWN"
                    outcome_prices_str = market.get('outcomePrices', '[]')
                    outcomes_str = market.get('outcomes', '[]')
                    
                    try:
                        # Parse the JSON strings
                        import json
                        outcome_prices = json.loads(outcome_prices_str)
                        outcomes = json.loads(outcomes_str)
                        
                        # Find which outcome has price 1 (the winning outcome)
                        if len(outcome_prices) == len(outcomes) == 2:
                            if outcome_prices[0] == "1":
                                resolution = outcomes[0].upper()  # First outcome won
                            elif outcome_prices[1] == "1":
                                resolution = outcomes[1].upper()  # Second outcome won
                            
                    except (json.JSONDecodeError, IndexError, TypeError) as e:
                        logger.warning(f"Error parsing outcome data for market {market.get('id', '')}: {e}")
                    
                    # Get market question and description
                    question = market.get('question', '')
                    description = market.get('description', '')
                    if description:
                        question = f"{question}\n{description}"
                    
                    return {
                        "source": "polymarket",
                        "url": f"https://polymarket.com/event/{market.get('slug', '')}",
                        "question": question,
                        "resolution": resolution,
                        "resolutionTime": end_timestamp
                    }
                
                valid_markets = [convert_polymarket_market(m) for m in markets if is_valid_market(m)]
                logger.info(f"Found {len(valid_markets)} valid markets at offset {offset}")

                offset += len(markets)
                if valid_markets:
                    yield valid_markets
                
                # Check if we should continue (if we're getting markets outside our date range)
                if markets:
                    last_market = markets[-1]
                    last_end_date_str = last_market.get('endDate')
                    if last_end_date_str:
                        try:
                            last_end_date = datetime.fromisoformat(last_end_date_str.replace('Z', '+00:00'))
                            last_end_timestamp = int(last_end_date.timestamp() * 1000)
                            if last_end_timestamp < START_TIMESTAMP:
                                logger.info(f"Reached markets outside date range at offset {offset}, stopping")
                                break
                        except ValueError:
                            pass
                
            except Exception as e:
                logger.error(f"Error fetching Polymarket offset {offset}: {e}")
                break


async def run_manifold():
    """Collect all Manifold market data."""
    logger.info("Starting Manifold data collection...")
    manifold_generator = ManifoldGenerator()
    manifold_data = []
    async for markets in manifold_generator:
        manifold_data.extend(markets)

    with open(f"manifold_markets_{START_DATE}_{END_DATE}.json", "w") as f:
        json.dump(manifold_data, f, indent=2)
    
    logger.info(f"Saved {len(manifold_data)} Manifold markets to file")
    return manifold_data


async def run_metaculus():
    """Collect all Metaculus question data."""
    logger.info("Starting Metaculus data collection...")
    metaculus_generator = MetaculusGenerator()
    metaculus_data = []
    async for questions in metaculus_generator:
        metaculus_data.extend(questions)

    with open(f"metaculus_questions_{START_DATE}_{END_DATE}.json", "w") as f:
        json.dump(metaculus_data, f, indent=2)
    
    logger.info(f"Saved {len(metaculus_data)} Metaculus questions to file")
    return metaculus_data


async def run_polymarket():
    """Collect all Polymarket market data."""
    logger.info("Starting Polymarket data collection...")
    polymarket_generator = PolymarketGenerator()
    polymarket_data = []
    async for markets in polymarket_generator:
        polymarket_data.extend(markets)

    with open(f"polymarket_markets_{START_DATE}_{END_DATE}.json", "w") as f:
        json.dump(polymarket_data, f, indent=2)
    
    logger.info(f"Saved {len(polymarket_data)} Polymarket markets to file")
    return polymarket_data


async def run_all_platforms(enable_manifold: bool = True, enable_metaculus: bool = True, enable_polymarket: bool = True):
    """Collect data from selected platforms."""
    logger.info("Starting data collection from selected platforms...")
    
    tasks = []
    manifold_data = []
    metaculus_data = []
    polymarket_data = []
    
    # Create tasks for enabled platforms
    if enable_manifold:
        logger.info("Manifold collection enabled")
        tasks.append(("manifold", asyncio.create_task(run_manifold())))
    else:
        logger.info("Manifold collection disabled")
    
    if enable_metaculus:
        logger.info("Metaculus collection enabled")
        tasks.append(("metaculus", asyncio.create_task(run_metaculus())))
    else:
        logger.info("Metaculus collection disabled")
    
    if enable_polymarket:
        logger.info("Polymarket collection enabled")
        tasks.append(("polymarket", asyncio.create_task(run_polymarket())))
    else:
        logger.info("Polymarket collection disabled")
    
    if not tasks:
        logger.warning("No platforms enabled! Please enable at least one platform.")
        return []
    
    # Wait for all enabled tasks to complete
    results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
    
    # Process results
    for i, (platform, _) in enumerate(tasks):
        result = results[i]
        if isinstance(result, Exception):
            logger.error(f"Error collecting data from {platform}: {result}")
        else:
            if platform == "manifold":
                manifold_data = result
            elif platform == "metaculus":
                metaculus_data = result
            elif platform == "polymarket":
                polymarket_data = result
    
    # Combine all data
    all_data = manifold_data + metaculus_data + polymarket_data
    
    # Save combined dataset if we have data from multiple platforms
    if len([t for t in tasks]) > 1 and all_data:
        with open(f"all_markets_{START_DATE}_{END_DATE}.json", "w") as f:
            json.dump(all_data, f, indent=2)
        logger.info(f"Saved combined dataset with {len(all_data)} total markets/questions")
    
    logger.info(f"Collection summary:")
    if enable_manifold:
        logger.info(f"  - Manifold: {len(manifold_data)} markets")
    if enable_metaculus:
        logger.info(f"  - Metaculus: {len(metaculus_data)} questions")
    if enable_polymarket:
        logger.info(f"  - Polymarket: {len(polymarket_data)} markets")
    logger.info(f"  - Total: {len(all_data)} items")
    
    return all_data


async def main():
    # You can now control which platforms to run by passing parameters
    await run_all_platforms(enable_manifold=True, enable_metaculus=True, enable_polymarket=True)

if __name__ == "__main__":
    asyncio.run(main())