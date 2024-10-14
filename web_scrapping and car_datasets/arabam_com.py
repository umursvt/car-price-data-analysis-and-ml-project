import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep

# Ana URL
base_url = 'https://www.arabam.com/ikinci-el/otomobil'


def get_car_data(page, take=50):
    # Dinamik URL yapısı
    url = f"{base_url}?take={take}&page={page}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.text
        soup = BeautifulSoup(data, 'html.parser')
        
        # Her ilanı içeren tr etiketlerini bul
        car_listings = soup.find_all('tr', class_='listing-list-item'.strip())
      
        cars_info =[]
        
        for car in car_listings:
            title = car.find('td',class_='listing-modelname pr').text.strip()
            # title dan bilgielri parse et
            title_parts = title.split(' ')
            brands = title_parts[0] if len(title_parts) > 0 else ' '
            models = title_parts[1] if len(title_parts) > 1 else ' '
            motors = title_parts[2] if len(title_parts) > 2 else ' '
            motor_infos = title_parts[3] if len(title_parts) > 3 else ' '
            motor_final = motors + ' ' + motor_infos
            # yıl fiyat renk bilgierini pars et
            years = car.find_all('td', class_='listing-text')[0].find('a').text.strip()
            kilometres = car.find_all('td', class_='listing-text')[1].find('a').text.strip()
            colors = car.find_all('td', class_='listing-text')[2].find('a').text.strip()
            advert_date = car.find_all('td',class_='listing-text')[3].find('a').text.strip()
            city = car.find_all('td',class_='listing-text')[4].find('a').text.strip()
            prices = car.find('span',class_='db no-wrap listing-price').text.strip().split(' ')[0]

            cars_info.append({
                'Advert Date':advert_date,
                'City':city,
                'Brand':brands,
                'Model':models,
                'Motor':motor_final,
                'Year':years,
                'Kilometres':kilometres,
                'Colors':colors,
                'Price':prices           
            })
        return cars_info

        
    else:
        print(f"Bağlantı hatası: {url}")
        return None  


all_pages= 50
all_car_data = []

for page in range(1,all_pages+1):
    page_car_data = get_car_data(page)
    sleep(1)
    if page_car_data:
        all_car_data.extend(page_car_data)
        
dataframe = pd.DataFrame(all_car_data)

dataframe.to_csv('web_scrapping and car_datasets/car_dataset_arabam_com_otomobil.csv')
    



    
