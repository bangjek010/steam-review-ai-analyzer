import steamreviews
import pandas as pd

def crawl_steam(app_id, review_type, language):
    request_params = {'language': language, 'review_type': review_type}
    review_dict, query_count = steamreviews.download_reviews_for_app_id(
        app_id, 
        chosen_request_params=request_params,
        verbose=False
    )
    
    reviews_data = review_dict['reviews']
    list_ulasan = []
    for r_id, r_info in reviews_data.items():
        list_ulasan.append({
            'Review_ID': r_id,
            'Review_Text': r_info['review'],
            'Playtime_Minutes': r_info['author']['playtime_forever'],
            'Review_Link': f"https://steamcommunity.com/profiles/{r_info['author']['steamid']}/recommended/{app_id}/"
        })
    return pd.DataFrame(list_ulasan)