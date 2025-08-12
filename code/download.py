import requests
from bs4 import BeautifulSoup
import os
import json
import time
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_session():
    """
    Create a configured request session
    """
    session = requests.Session()
    
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=10,    
        pool_maxsize=10,        
        max_retries=3,                    
        pool_block=False                  
    )
    
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (compatible; VideoBot/1.0; Custom Video Downloader)',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate',
    })
    
    return session

def check_ffmpeg():
    """Check if ffmpeg is installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        logger.error("Error: ffmpeg is not installed. Please install ffmpeg for video format conversion.")
        return False

def convert_to_mp4(input_file, output_file):
    """Convert webm format to mp4 format"""
    try:
        subprocess.run([
            'ffmpeg', 
            '-i', input_file,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-y',
            output_file
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.remove(input_file) 
        return True
    except Exception as e:
        logger.error(f"Error converting video format: {e}")
        if os.path.exists(output_file):
            os.remove(output_file)
        return False

def download_video(session, url, filepath):
    """Download video file"""
    try:
        logger.info(f"Starting video download: {url}")
        response = session.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\rDownload progress: {progress:.1f}%", end='', flush=True)
        
        print()  # New line
        logger.info(f"Video download completed: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def process_single_date(session, date_str, index, total):
    """Process video for a single date"""
    temp_webm_path = os.path.join('temp', f'{date_str}.webm')
    final_mp4_path = os.path.join('videos', f'{date_str}.mp4')

    # Check if file already exists
    if os.path.exists(final_mp4_path):
        logger.info(f"Skipping existing file: {date_str} ({index}/{total})")
        return True

    try:
        url = f"https://commons.wikimedia.org/wiki/Template:Motd/{date_str}"
        logger.info(f"Processing {date_str} ({index}/{total})...")
        
        response = session.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        video_element = soup.find('video')

        if not video_element:
            logger.warning(f"Video element not found: {date_str}")
            return False

        sources = video_element.find_all('source')
        video_urls = {}
        for source in sources:
            quality = source.get('data-width', '0')
            video_url = source.get('src', '')
            if quality.isdigit() and video_url:
                if not video_url.startswith('http'):
                    video_url = 'https:' + video_url
                video_urls[int(quality)] = video_url

        if not video_urls:
            logger.warning(f"Video URL not found: {date_str}")
            return False

        # Select highest quality video
        best_quality = max(video_urls.keys())
        video_url = video_urls[best_quality]
        logger.info(f"Selected quality: {best_quality}p")

        # Download video
        if download_video(session, video_url, temp_webm_path):
            logger.info(f"Converting {date_str} to MP4 format...")
            if convert_to_mp4(temp_webm_path, final_mp4_path):
                logger.info(f"✅ Successfully downloaded and converted {date_str}")
                return True
            else:
                logger.error(f"❌ Conversion failed: {date_str}")
        else:
            logger.error(f"❌ Download failed: {date_str}")

    except Exception as e:
        logger.error(f"Error processing {date_str}: {e}")
        # Clean up temporary files
        if os.path.exists(temp_webm_path):
            os.remove(temp_webm_path)
    
    return False

def load_dates_from_json(json_file_path):
    """Load date list from JSON file"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            logger.info(f"File size: {len(content)} characters")
            logger.info(f"First 200 characters: {content[:200]}")
            
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"JSON file structure: {type(data)}")
        
        dates = []
        if isinstance(data, list):
            # If it's an array format
            logger.info(f"Processing array with {len(data)} items")
            for i, item in enumerate(data[:5]):  # Show first 5 items for debugging
                logger.info(f"Item {i}: {item}")
                if isinstance(item, dict):
                    if 'date' in item:
                        date_value = item['date']
                        logger.info(f"Item {i}: date field = '{date_value}' (type: {type(date_value)})")
                        if isinstance(date_value, str) and date_value.strip():
                            dates.append(date_value.strip())
                        else:
                            logger.warning(f"Item {i}: Invalid date value - {date_value} (type: {type(date_value)})")
                    else:
                        logger.warning(f"Item {i}: Missing 'date' key. Available keys: {list(item.keys())}")
                else:
                    logger.warning(f"Item {i}: Not a dictionary - {type(item)}")
            
            # Continue processing all items (without detailed logging)
            for item in data:
                if isinstance(item, dict) and 'date' in item:
                    date_value = item['date']
                    if isinstance(date_value, str) and date_value.strip():
                        dates.append(date_value.strip())
                        
        elif isinstance(data, dict) and 'date' in data:
            # If it's a single object
            date_value = data['date']
            logger.info(f"Single object: date field = '{date_value}' (type: {type(date_value)})")
            if isinstance(date_value, str) and date_value.strip():
                dates.append(date_value.strip())
            else:
                logger.warning(f"Single object: Invalid date value - {date_value} (type: {type(date_value)})")
        else:
            logger.error(f"Unexpected JSON structure or missing 'date' key")
            if isinstance(data, dict):
                logger.error(f"Available keys: {list(data.keys())}")
        
        # Remove duplicates while preserving order
        unique_dates = list(dict.fromkeys(dates))
        
        logger.info(f"Successfully loaded {len(unique_dates)} unique valid dates from {json_file_path}")
        if unique_dates:
            logger.info(f"First few dates: {unique_dates[:5]}")
        
        return unique_dates
    
    except Exception as e:
        logger.error(f"Error reading JSON file: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []

def scrape_videos_from_json(json_file_path):
    """Download videos in single thread based on dates from JSON file"""
    if not check_ffmpeg():
        return 0

    # Create necessary directories
    for directory in ['videos', 'temp']:
        os.makedirs(directory, exist_ok=True)

    # Load dates from JSON file
    dates = load_dates_from_json(json_file_path)
    if not dates:
        logger.error("Failed to get valid dates from JSON file")
        return 0

    logger.info(f"Processing first {len(dates)} dates for testing")

    # Create session
    session = create_session()
    success_count = 0
    
    try:
        for i, date_str in enumerate(dates, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting to process {i}/{len(dates)} date: {date_str}")
            logger.info(f"{'='*50}")
            
            if process_single_date(session, date_str, i, len(dates)):
                success_count += 1
            
            # Add delay to avoid too frequent requests
            if i < len(dates):  # Not the last one
                logger.info("Waiting 2 seconds before continuing...")
                time.sleep(2)
                
    finally:
        session.close()

    # Clean up temporary directory
    try:
        for file in os.listdir('temp'):
            os.remove(os.path.join('temp', file))
        os.rmdir('temp')
        logger.info("Temporary files cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {e}")

    return success_count

def validate_files():
    """Check video files existence"""
    videos_dir = 'videos'
    
    if not os.path.exists(videos_dir):
        logger.warning("Videos directory does not exist")
        return False
    
    videos = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
    logger.info(f"Total video files found: {len(videos)}")
        
    return len(videos) > 0

def main():
    """Main function"""
    json_file_path = '../data/VideoSimpleQA.json'  # JSON file path, modify as needed
    
    if not os.path.exists(json_file_path):
        logger.error(f"JSON file does not exist: {json_file_path}")
        logger.info("Please ensure the JSON file exists with date information in the following format:")
        logger.info('[{"date": "2009-06-02", ...}, {"date": "2009-06-03", ...}]')
        return
    
    logger.info(f"Starting to read dates from {json_file_path} and download videos...")
    
    total_downloaded = scrape_videos_from_json(json_file_path)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Download completed! Successfully downloaded {total_downloaded} files")
    logger.info(f"{'='*60}")
    
    if validate_files():
        logger.info("✅ All video files downloaded successfully!")
    else:
        logger.warning("⚠️  No video files found! Please check the logs for details.")

if __name__ == '__main__':
    main()