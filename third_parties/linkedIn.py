import os 
import requests
from config.tp_secrets import Secrets
import json



def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool=False):
    """scrape information from LinkedIn profiles,
    Manually scrape the information from the LinkedIn profile"""

    os.environ['PROXYCURL_API_KEY'] = Secrets.proxy_curl_api_key

    if mock:
        with open('third_parties/mock_linkedIn.json', 'r') as f:
            data = json.load(f)
    else:
        api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
        head_dic = {"Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY")}'}
        response = requests.get(
            api_endpoint,
            params={"url": linkedin_profile_url},
            headers=head_dic,
            timeout=10
        )
        data = response.json()
    
    # remove key with empty values
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")
    
    return data


if __name__=="__main__":
    print(scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/eden-marco/", mock=True))