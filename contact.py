import requests
import time
import csv
from flask import Flask, request, jsonify, send_file
import io
import re
import google.generativeai as genai
import concurrent.futures
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import defaultdict, namedtuple
from datetime import datetime
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)


genai_api_key = "AIzaSyD3FfZW4oj1w1Q9T5aCUi_QZmeLJInkkqE"
API_KEY = 'AIzaSyDrLHe1dTTc78XG7lznF0fWF0o5uKR6HXA'
genai.configure(api_key=genai_api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

EXCLUDED_WEBSITES = [
    'tradeindia.com',
    'alibaba.com',
    'made-in-china.com',
    'exportersindia.com',
    'justdial.com',
    'sulekha.com',
    'yellowpages.com',
    'yellowpages.in',
    'google.com',
    'facebook.com',
    'linkedin.com',
    'twitter.com',
    'instagram.com',
    'youtube.com'
]

def fetch_website_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            text = resp.text
            text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'\s+', ' ', text)
            return text[:3000], 'Reachable'
        else:
            logging.warning(f"Failed to fetch {url}: Status {resp.status_code}")
            return '', 'Unreachable'
    except requests.exceptions.Timeout:
        logging.error(f"Timeout fetching website content for {url}")
        return '', 'Timeout'
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching website content for {url}: {e}")
        return '', 'Unreachable'
    except Exception as e:
        logging.error(f"Unexpected error in fetch_website_text for {url}: {e}")
        return '', 'Error'


def analyze_buyer_potential(company_name, company_address, website, phone, industry, product_info):
    """
    Use Gemini API to analyze buyer probability based on company info, product, and website content
    """
    try:
        logging.info(f"Analyzing buyer potential for {company_name} ({website})")
        website_text, website_status = fetch_website_text(website) if website else ('', 'No Website')

        prompt = f"""
You're an AI sales assistant helping qualify B2B leads.

Analyze if this company is a potential buyer for the product/service described.

---

**Company Info**
- Name: {company_name}
- Address: {company_address}
- Website: {website}
- Phone: {phone}
- Industry: {industry}

**Product/Service**:
{product_info}

**Website Content (homepage snippet)**:
{website_text}

---

Respond in this structured format:

BUYER PROBABILITY: [0-100, where 100 = very likely to buy]
REASON: [1-2 specific reasons why this company would (or would not) buy. Mention alignment with product, relevant offerings, keywords, etc.]
"""
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        prob_match = re.search(r'BUYER PROBABILITY:\s*(\d{1,3})', response_text, re.IGNORECASE)
        reason_match = re.search(r'REASON:\s*(.+)', response_text, re.IGNORECASE | re.DOTALL)

        if prob_match:
            probability = int(prob_match.group(1))
            probability = min(max(probability, 0), 100)  # Clamp to 0-100
            reason = reason_match.group(1).strip() if reason_match else "Analysis completed, reason not explicitly parsed."
            result = {
                'Buyer Probability': probability,
                'Reason': reason,
                'Source': 'Source for probability: Website content + company profile'
            }
            result['Website Status'] = website_status
            return result
        else:
            logging.warning(f"Could not parse Gemini response for {company_name}: {response_text[:200]}...")
            result = {
                'Buyer Probability': 0, # Default to 0 if unable to parse
                'Reason': 'Unable to parse AI analysis - default assigned.',
                'Source': 'Source for probability: Website content + company profile'
            }
            result['Website Status'] = website_status
            return result
    except genai.types.BlockedPromptException as e:
        logging.error(f"Gemini API blocked prompt for {company_name}: {e}")
        result = {
            'Buyer Probability': 0,
            'Reason': f'AI analysis blocked: {e}',
            'Source': 'Source for probability: Website content + company profile'
        }
        result['Website Status'] = website_status
        return result
    except Exception as e:
        logging.error(f"Error analyzing buyer probability for {company_name}: {e}")
        result = {
            'Buyer Probability': 0, # Default to 0 if analysis fails
            'Reason': f'AI analysis failed: {e}',
            'Source': 'Source for probability: Website content + company profile'
        }
        result['Website Status'] = website_status
        return result

def get_rating_color(rating):
    if rating >= 85:
        return 'rating-excellent'
    elif rating >= 65:
        return 'rating-good'
    elif rating >= 45:
        return 'rating-fair'
    else:
        return 'rating-poor'

def get_rating_label(rating):
    if rating >= 85:
        return 'Hot Lead'
    elif rating >= 65:
        return 'Warm Lead'
    elif rating >= 45:
        return 'Cold Lead'
    else:
        return 'Poor Match'

def is_valid_website(website):
    if not website or website.strip() == '':
        return False

    website_lower = website.lower()

    try:
        parsed_url = urlparse(website_lower)
        if not parsed_url.scheme in ['http', 'https']:
            return False
        if not parsed_url.netloc:
            return False

        for excluded_site in EXCLUDED_WEBSITES:
            if excluded_site in parsed_url.netloc or parsed_url.netloc.endswith('.' + excluded_site):
                return False
    except ValueError:
        return False

    return True

def search_places(query, location, region=None, radius=50000):
    endpoint = 'https://maps.googleapis.com/maps/api/place/textsearch/json'
    if region:
        search_location = f"{region}, {location}"
    else:
        search_location = location

    params = {
        'query': f"{query} in {search_location}",
        'key': API_KEY,
        'type': 'business',
        'radius': radius,
        'language': 'en',
        'rankby': 'prominence'
    }

    results = []
    page_count = 0
    max_pages = 2

    try:
        while True and page_count < max_pages:
            logging.info(f"Fetching Places API page {page_count + 1} for query: {query} in {search_location}")
            response = requests.get(endpoint, params=params, timeout=15)
            response.raise_for_status()
            res = response.json()

            if res.get('status') == 'REQUEST_DENIED':
                logging.error(f"Places API Request Denied: {res.get('error_message', 'Unknown error')}. Check API key and billing.")
                break
            elif res.get('status') != 'OK' and res.get('status') != 'ZERO_RESULTS':
                logging.warning(f"Places API Error: {res.get('status')} - {res.get('error_message', 'Unknown error')}")
                break

            new_results = res.get('results', [])
            if not new_results:
                break

            results.extend(new_results)
            page_count += 1

            if 'next_page_token' not in res:
                break

            time.sleep(1)
            params = {
                'pagetoken': res['next_page_token'],
                'key': API_KEY
            }
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error during Places search: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in search_places: {e}")

    return results

def get_place_details(place_id):
    endpoint = 'https://maps.googleapis.com/maps/api/place/details/json'
    params = {
        'place_id': place_id,
        'fields': 'name,formatted_address,website,formatted_phone_number,rating,user_ratings_total,business_status,opening_hours,types,price_level,reviews,photos,international_phone_number,geometry,address_components',
        'key': API_KEY,
        'language': 'en'
    }

    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        res = response.json()

        if res.get('status') == 'REQUEST_DENIED':
            logging.error(f"Places Details API Request Denied: {res.get('error_message', 'Unknown error')}. Check API key and billing.")
            return {}
        if res.get('status') != 'OK':
            logging.warning(f"Places Details API Error: {res.get('status')} - {res.get('error_message', 'Unknown error')}")
            return {}

        return res.get('result', {})
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error in place details for {place_id}: {e}")
        return {}
    except Exception as e:
        logging.error(f"Unexpected error in get_place_details for {place_id}: {e}")
        return {}

def is_valid_url(url, base_domain):
    parsed = urlparse(url)
    if parsed.scheme in ['http', 'https'] and \
       (parsed.netloc == base_domain or parsed.netloc.endswith('.' + base_domain)):
        return True
    return False

def extract_emails_from_text(text):
    potential_emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    valid_emails = []
    for email in potential_emails:
        if re.search(r'\.(jpg|jpeg|png|gif|webp|svg|bmp)(@|$)', email, re.IGNORECASE):
            continue
        if not re.search(r'\.[a-zA-Z]{2,6}$', email, re.IGNORECASE):
            continue
        valid_emails.append(email)
    return valid_emails

def classify_email(email):
    email_lower = email.lower()
    if any(keyword in email_lower for keyword in ['sales', 'business', 'bd', 'enquiry', 'inquiry', 'deal']):
        return 'Sales/Business'
    elif any(keyword in email_lower for keyword in ['hr', 'careers', 'jobs', 'recruit']):
        return 'HR'
    elif any(keyword in email_lower for keyword in ['support', 'help', 'info', 'service', 'contact']):
        return 'Support/Info'
    elif any(keyword in email_lower for keyword in ['admin', 'webmaster', 'abuse']):
        return 'Admin'
    else:
        return 'General'

EmailInfo = namedtuple('EmailInfo', ['email', 'source'])

def extract_emails_from_text_with_source(text, source):
    emails = extract_emails_from_text(text)
    return [EmailInfo(email, source) for email in emails]

async def fetch_url(session, url, semaphore):
    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status == 200:
                    content = await response.text()
                    return url, content
                else:
                    logging.debug(f"Failed to fetch {url} with status {response.status}")
                    return url, None
        except aiohttp.client_exceptions.ClientConnectorError as e:
            logging.debug(f"Client connection error for {url}: {e}")
            return url, None
        except asyncio.TimeoutError:
            logging.debug(f"Timeout fetching {url}")
            return url, None
        except Exception as e:
            logging.debug(f"Error fetching {url}: {e}")
            return url, None

async def process_url_with_source(session, url, base_domain, semaphore, visited, emails_found, source_label):
    if url in visited:
        return []
    visited.add(url)
    url_fetched, content = await fetch_url(session, url, semaphore)
    if not content:
        return []
    emails = extract_emails_from_text_with_source(content, source_label)
    emails_found.extend(emails)

    soup = BeautifulSoup(content, 'html.parser')
    new_links = []

    priority_keywords = ['contact', 'about', 'team', 'support', 'help', 'inquiry', 'enquiry', 'careers', 'jobs']

    for link in soup.find_all('a', href=True):
        full_url = urljoin(url_fetched, link['href']).split('#')[0].split('?')[0]
        link_text = link.get_text(strip=True).lower()

        if full_url not in visited and is_valid_url(full_url, base_domain):
            if 'mailto:' in full_url:
                continue

            if any(keyword in link_text for keyword in priority_keywords):
                new_links.append((full_url, 'priority_link'))
            elif source_label == 'homepage':
                new_links.append((full_url, 'general_link'))
    return new_links

async def extract_emails_from_website_async_smart(base_url, depth=1, max_concurrent=10):
    visited = set()
    emails_found = []
    base_domain = urlparse(base_url).netloc

    if not base_domain:
        logging.warning(f"Invalid base URL for email extraction: {base_url}")
        return defaultdict(lambda: defaultdict(list))

    to_visit_current_depth = [(base_url, 'homepage')]

    connector = aiohttp.TCPConnector(
        limit=max_concurrent,
        limit_per_host=min(max_concurrent, 5),
        ttl_dns_cache=300,
        verify_ssl=False
    )
    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession(
        connector=connector,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'}
    ) as session:
        for current_d in range(depth):
            if not to_visit_current_depth:
                break

            logging.info(f"Crawling depth {current_d+1} for {base_url}. Visiting {len(to_visit_current_depth)} URLs.")

            tasks = [process_url_with_source(session, url, base_domain, semaphore, visited, emails_found, source_label)
                     for url, source_label in to_visit_current_depth]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            to_visit_next_depth = []
            for result in results:
                if isinstance(result, list):
                    for link, link_type in result:
                        if link not in visited:
                            to_visit_next_depth.append((link, link_type))
                elif isinstance(result, Exception):
                    logging.error(f"Error during async URL processing: {result}")

            if depth > 1 and current_d == 0:
                to_visit_current_depth = [
                    (link, 'subpage') for link, link_type in to_visit_next_depth
                ]
            else:
                to_visit_current_depth = []

    grouped_emails = defaultdict(lambda: defaultdict(list))
    unique_emails_set = set()

    for email_info in emails_found:
        if email_info.email not in unique_emails_set:
            unique_emails_set.add(email_info.email)
            category = classify_email(email_info.email)
            grouped_emails[category][email_info.source].append(email_info.email)

    for category in grouped_emails:
        for source in grouped_emails[category]:
            grouped_emails[category][source] = sorted(list(set(grouped_emails[category][source])))

    return grouped_emails

def extract_emails_from_website_smart(base_url, depth=1):
    if not base_url:
        return {}
    try:
        logging.info(f"Starting email extraction for {base_url} at depth {depth}")
        emails = asyncio.run(extract_emails_from_website_async_smart(base_url, depth))
        logging.info(f"Finished email extraction for {base_url}.")
        return emails
    except Exception as e:
        logging.error(f"Error extracting emails from website {base_url}: {e}")
        return {}


# Moved analyze_wrapper to the top level
def analyze_wrapper(company_data):
    """
    Wrapper function to perform lead analysis and email extraction for a single company,
    designed to be called by ThreadPoolExecutor.
    """
    analysis = analyze_buyer_potential(
        company_data['name'],
        company_data['address'],
        company_data['website'],
        company_data['phone'],
        company_data['industry'],
        company_data['product_info']
    )
    website_status = analysis.get('Website Status', 'Reachable')

    buyer_probability = int(analysis.get('Buyer Probability', 0))

    adjusted_probability = buyer_probability

    if company_data['google_rating'] > 0:
        google_rating_factor = (company_data['google_rating'] / 5) * 100 * 0.2
        adjusted_probability = int(adjusted_probability * 0.8 + google_rating_factor)

    if company_data['rating_count'] > 100:
        adjusted_probability += 5
    elif company_data['rating_count'] > 50:
        adjusted_probability += 3
    elif company_data['rating_count'] > 10:
        adjusted_probability += 1

    relevance_boost = min(company_data['relevance_score'] / 10, 10)
    adjusted_probability += relevance_boost

    adjusted_probability = max(0, min(100, adjusted_probability))

    emails_display = 'Not available' # Default text if no emails found or website unreachable

    if website_status == 'Unreachable' or website_status == 'Timeout':
        emails_display = f'Website {website_status.lower()}, cannot extract emails'
    elif website_status == 'No Website':
        emails_display = 'No website provided for this company'
    else:
        emails_grouped = extract_emails_from_website_smart(company_data['website'], depth=2) if company_data['website'] else {}
        if not emails_grouped:
            emails_display = 'No email provided by owner or found on website'
        else:
            unique_emails_set = set() # To ensure globally unique emails in the final string
            parts = []
            for dept, sources in emails_grouped.items():
                for src, emails in sources.items():
                    current_source_emails = []
                    for email in emails:
                        if email not in unique_emails_set:
                            unique_emails_set.add(email)
                            current_source_emails.append(email)
                    if current_source_emails:
                        label = f"{dept.capitalize()} ({src.replace('_', ' ').capitalize()})"
                        parts.append(f"{label}: {', '.join(sorted(current_source_emails))}")
            emails_display = '; '.join(parts) if parts else 'No email provided by owner or found on website'


    top_review = ''
    if company_data['reviews'] and len(company_data['reviews']) > 0:
        top_review = company_data['reviews'][0].get('text', '')[:200]
        if len(company_data['reviews'][0].get('text', '')) > 200:
            top_review += '...'

    opening_hours_text = 'Not available'
    if 'opening_hours' in company_data and 'weekday_text' in company_data['opening_hours']:
        opening_hours_text = '\n'.join(company_data['opening_hours']['weekday_text'])

    lead_data = {
        'Company Name': company_data['name'],
        'Address': company_data['address'],
        'Website': company_data['website'],
        'Phone Number': company_data['phone'],
        'Business Type': company_data['business_type'] if company_data['business_type'] != 'all' else 'Auto-detected',
        'Buyer Probability': adjusted_probability,
        'Original Probability': buyer_probability,
        'Google Rating': f"{company_data['google_rating']}/5 ({company_data['rating_count']} reviews)" if company_data['google_rating'] else 'No ratings',
        'Business Status': company_data['business_status'],
        'Place Types': ', '.join(company_data['place_types'][:5]) if company_data['place_types'] else '',
        'Top Review': top_review,
        'Reason': analysis.get('Reason', ''),
        'Source': analysis.get('Source', ''),
        'Website Status': website_status,
        'Emails': emails_display,
        'Rating Color': get_rating_color(adjusted_probability),
        'Rating Label': get_rating_label(adjusted_probability),
        'Latitude': company_data['latitude'],
        'Longitude': company_data['longitude'],
        'Opening Hours': opening_hours_text
    }
    return lead_data


def lead_generation(industry, location, business_type='all', region=None, radius=50000, product_info=''):
    max_leads = 10

    if not industry or not location:
        logging.error("Industry and location are required for lead generation.")
        return []

    search_query = f"{industry} business"
    if business_type and business_type != 'all':
        search_query = f"{industry} {business_type}"

    logging.info(f"Starting lead generation for query: '{search_query}' in '{location}'")
    raw_places = search_places(search_query, location, region, radius)

    if not raw_places:
        logging.warning("No places found in initial search. Trying a more generic query.")
        search_query_fallback = industry
        raw_places = search_places(search_query_fallback, location, region, radius)
        if not raw_places:
            logging.warning("No places found even with generic query. Returning empty leads.")
            return []

    companies_to_analyze_candidates = []
    seen_place_ids = set()

    logging.info(f"Found {len(raw_places)} raw places from Google Maps. Filtering and getting details...")

    place_ids_to_fetch = []
    for place in raw_places:
        place_id = place.get('place_id')
        if place_id and place_id not in seen_place_ids:
            seen_place_ids.add(place_id)
            place_ids_to_fetch.append(place_id)
        if len(place_ids_to_fetch) >= max_leads * 3:
            break

    detailed_places = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(get_place_details, pid): pid for pid in place_ids_to_fetch}
        for future in concurrent.futures.as_completed(futures):
            place_id = futures[future]
            try:
                details = future.result()
                if details:
                    detailed_places.append(details)
            except Exception as e:
                logging.error(f"Error fetching details for place_id {place_id}: {e}")

    logging.info(f"Fetched details for {len(detailed_places)} places. Applying relevance filters.")

    for details in detailed_places:
        name = details.get('name')
        address = details.get('formatted_address')
        website = details.get('website', '')
        phone = details.get('formatted_phone_number', '')
        international_phone = details.get('international_phone_number', '')
        google_rating = details.get('rating', 0)
        rating_count = details.get('user_ratings_total', 0)
        business_status = details.get('business_status', '')
        place_types = details.get('types', [])
        reviews = details.get('reviews', [])
        geometry = details.get('geometry', {})
        location_data = geometry.get('location', {})
        latitude = location_data.get('lat', 0)
        longitude = location_data.get('lng', 0)
        opening_hours = details.get('opening_hours', {})

        if not name or not address:
            continue
        if not phone and not international_phone:
            continue
        if not phone and international_phone:
            phone = international_phone

        if business_status == 'PERMANENTLY_CLOSED':
            continue

        if not is_valid_website(website):
            continue

        relevance_score = 0
        name_lower = name.lower()
        industry_lower = industry.lower()

        if industry_lower in name_lower:
            relevance_score += 30

        for btype in place_types:
            if industry_lower in btype.lower():
                relevance_score += 25
            if btype in ['store', 'establishment', 'business', 'company', 'office', 'service']:
                relevance_score += 10

        if website:
            relevance_score += 15

        if rating_count > 0:
            relevance_score += min(rating_count / 10, 15)

        if ('food' not in industry_lower and 'restaurant' not in industry_lower and
            'cafe' not in industry_lower and 'catering' not in industry_lower):
            if any(food_type in place_types for food_type in ['restaurant', 'food', 'cafe', 'meal_delivery', 'meal_takeaway']):
                relevance_score -= 50
            if any(food_term in name_lower for food_term in ['restaurant', 'food', 'cafe', 'chicken', 'pizza', 'burger']):
                relevance_score -= 30

        if relevance_score < 20:
            continue

        companies_to_analyze_candidates.append({
            'name': name,
            'address': address,
            'website': website,
            'phone': phone,
            'industry': industry,
            'product_info': product_info,
            'business_type': business_type,
            'google_rating': google_rating,
            'rating_count': rating_count,
            'business_status': business_status,
            'place_types': place_types,
            'reviews': reviews,
            'latitude': latitude,
            'longitude': longitude,
            'opening_hours': opening_hours,
            'relevance_score': relevance_score
        })

    companies_to_analyze_candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
    companies_to_analyze = companies_to_analyze_candidates[:max_leads]

    logging.info(f"Selected {len(companies_to_analyze)} companies for detailed buyer potential analysis.")

    leads = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(companies_to_analyze), 5)) as executor:
        futures = {executor.submit(analyze_wrapper, company_data): company_data['name'] for company_data in companies_to_analyze}

        for future in concurrent.futures.as_completed(futures):
            company_name = futures[future]
            try:
                lead_data = future.result()
                if lead_data:
                    leads.append(lead_data)
                    logging.info(f"Successfully analyzed {company_name}. Probability: {lead_data.get('Buyer Probability')}")
            except Exception as e:
                logging.error(f"Error during analysis for {company_name}: {e}")

    if leads:
        leads.sort(key=lambda x: int(x.get('Buyer Probability', 0)), reverse=True)
        logging.info(f"Finished lead generation. Found {len(leads)} qualified leads.")
    else:
        logging.warning("No qualified leads found after analysis.")

    return leads

def suggest_industry_location(product):
    try:
        logging.info(f"Requesting suggestions for product: {product}")
        prompt = f"""
Based on the following product/service description, suggest the most relevant target industries and locations for B2B lead generation:

Product/Service: {product}

Provide your response in this exact format:

SUGGESTED INDUSTRIES: [List 3-5 specific industries that would be most interested in this product/service, separated by commas]
SUGGESTED LOCATIONS: [List 3-5 specific cities or regions in India where demand for this product/service would be highest, separated by commas]
REASONING: [Brief explanation of why these industries and locations are good targets]
"""
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        industries_match = re.search(r'SUGGESTED INDUSTRIES:\s*(.+?)\n', response_text, re.IGNORECASE | re.DOTALL)
        locations_match = re.search(r'SUGGESTED LOCATIONS:\s*(.+?)\n', response_text, re.IGNORECASE | re.DOTALL)
        reasoning_match = re.search(r'REASONING:\s*(.+)', response_text, re.IGNORECASE | re.DOTALL)

        industries = industries_match.group(1).strip() if industries_match else ""
        locations = locations_match.group(1).strip() if locations_match else ""
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        logging.info(f"Generated suggestions: Industries='{industries}', Locations='{locations}'")
        return {
            'industries': industries,
            'locations': locations,
            'reasoning': reasoning
        }
    except genai.types.BlockedPromptException as e:
        logging.error(f"Gemini API blocked suggestion prompt: {e}")
        return {
            'industries': "",
            'locations': "",
            'reasoning': f"AI analysis blocked: {e}"
        }
    except Exception as e:
        logging.error(f"Error generating suggestions: {e}")
        return {
            'industries': "",
            'locations': "",
            'reasoning': f"Unable to generate suggestions at this time: {e}"
        }

@app.route('/suggest', methods=['POST'])
def get_suggestions():
    product = request.json.get('product', '')
    if not product:
        logging.warning("Missing 'product' in /suggest request.")
        return jsonify({'error': 'Product description is required'}), 400

    try:
        suggestions = suggest_industry_location(product)
        return jsonify(suggestions)
    except Exception as e:
        logging.exception("An error occurred during suggestion generation.")
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500

@app.route('/generate', methods=['POST'])
def generate_leads():
    data = request.json
    industry = data.get('industry')
    location = data.get('location')
    product_info = data.get('product_info', '')

    logging.info(f"Received generate request: Industry='{industry}', Location='{location}'")

    if not industry or not location:
        logging.warning("Missing industry or location in generate request.")
        return jsonify({'error': 'Industry and location are required fields'}), 400

    try:
        leads = lead_generation(industry, location, product_info=product_info)
        logging.info(f"Returning {len(leads)} leads to client.")
        return jsonify(leads)
    except Exception as e:
        logging.exception("An error occurred during lead generation.")
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500

@app.route('/download', methods=['POST'])
def download_csv():
    leads = request.json.get('leads', [])
    output = io.StringIO()

    # CORRECTED LINE HERE
    base_fields = ['Company Name', 'Address', 'Website', 'Phone Number', 'Business Type', 'Google Rating', 'Business Status']
    analysis_fields = ['Buyer Probability', 'Reason', 'Source', 'Website Status', 'Emails', 'Rating Label', 'Opening Hours']
    location_fields = ['Latitude', 'Longitude']

    fieldnames = base_fields + analysis_fields + location_fields
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    csv_data = []
    for lead in leads:
        csv_row = {field: lead.get(field, '') for field in fieldnames}
        csv_data.append(csv_row)

    writer.writerows(csv_data)
    mem = io.BytesIO()
    mem.write(output.getvalue().encode('utf-8'))
    mem.seek(0)
    output.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    download_name = f'leads_export_{timestamp}.csv'

    return send_file(mem, mimetype='text/csv', as_attachment=True, download_name=download_name)

if __name__ == '__main__':
    logging.info("Starting Flask application.")
    app.run(debug=True)