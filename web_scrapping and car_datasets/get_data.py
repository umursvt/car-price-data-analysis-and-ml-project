import requests
from bs4 import BeautifulSoup

# Ana url yapısı
base_url = 'https://www.arabam.com/ikinci-el/arazi-suv-pick-up'

def get_car_data(page, take=50):
    # URL'yi dinamik olarak sayfa ve take parametreleriyle oluşturma
    url = f"{base_url}?take={take}&page={page}"
    
    # Veri getir
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.text
        soup = BeautifulSoup(data, 'html.parser')
        
        # Her ilanı içeren tr etiketlerini bul
        car_listings = soup.find_all('tr', class_='listing-list-item should-hover bg-white')
        
        cars = []
        
        # Her satırdaki ilan bilgilerini işleme
        for car in car_listings:
            title = car.find('td', class_='listing-modelname pr').text.strip()
            year = car.find_all('td')[4].text.strip()  # Model yılı
            km = car.find_all('td')[5].text.strip()    # KM bilgisi
            color = car.find_all('td')[6].text.strip() # Renk
            
            # Fiyat bilgisi bulunamazsa None dönmemesi için kontrol ediyoruz
            price_element = car.find('td', class_='listing-price')
            if price_element:
                price = price_element.text.strip()
            else:
                price = "Fiyat bilgisi yok"

            cars.append({
                'Title': title,
                'Year': year,
                'KM': km,
                'Color': color,
                'Price': price
            })
        
        return cars
    
    else:
        print(f"Sayfa {page} yüklenemedi.")
        return []

# Tüm sayfalardan verileri çekme
all_car_names = []
total_pages = 100  # 50 sayfa olduğunu varsayıyoruz

for page in range(1, total_pages + 1):
    car_names = get_car_data(page)
    all_car_names.extend(car_names)  # Her sayfanın verilerini listeye ekliyoruz

# Sonuçları yazdırma
for car in all_car_names:
    print(f"Model: {car['Title']}")
    print(f"Yıl: {car['Year']}")
    print(f"Kilometre: {car['KM']}")
    print(f"Renk: {car['Color']}")
    print(f"Fiyat: {car['Price']}")
    print("-" * 40)

# Kaç tane veri çekildiğini gösterme
print(f"Toplam {len(all_car_names)} araç bilgisi çekildi.")
