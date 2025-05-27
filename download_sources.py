import os
import asyncio
import httpx

LOCAL_DOCUMENT_MAP = {
    "https://help.typeform.com/hc/en-us/articles/23541138531732-Create-multi-language-forms": "data/multi_language_forms.txt",
    "https://help.typeform.com/hc/en-us/articles/27703634781076-Add-a-Multi-Question-Page-to-your-form": "data/multi_question_page.txt"
}

async def download_html_files():
    """
    Downloads raw HTML content from specified URLs and saves them to local files
    in the 'data' directory as per the LOCAL_DOCUMENT_MAP.
    """
    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True) # Create the data folder if it doesn't exist

    async with httpx.AsyncClient() as client:
        # Add a User-Agent header to mimic a browser, which can help bypass 403 errors
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36"
        }

        for url, file_path in LOCAL_DOCUMENT_MAP.items():
            print(f"Attempting to download from: {url}")
            try:
                response = await client.get(url, timeout=30.0, headers=headers)
                response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

                # Ensure the directory for the file exists if it's nested
                file_dir = os.path.dirname(file_path)
                if file_dir and file_dir != data_folder: # Only create if it's a sub-directory
                    os.makedirs(file_dir, exist_ok=True)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"Successfully downloaded {url} to {file_path}")

            except httpx.RequestError as e:
                print(f"Error downloading {url}: Network error - {e}")
            except httpx.HTTPStatusError as e:
                print(f"Error downloading {url}: HTTP error - {e.response.status_code} {e.response.reason_phrase}")
            except Exception as e:
                print(f"An unexpected error occurred while processing {url}: {e}")

if __name__ == "__main__":
    asyncio.run(download_html_files())