from playwright.async_api import async_playwright
from dataclasses import dataclass, field, asdict
from typing import List
import json

@dataclass
class Business:
    name: str = None
    address: str = None
    website: str = None
    rating: float = None
    top_reviews: List[str] = field(default_factory=list)

async def scrape_google_maps(service: str, city: str) -> List[Business]:
    businesses = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto("https://www.google.com/maps", timeout=80000)
        await page.wait_for_selector('//input[@id="searchboxinput"]', timeout=10000)

        # Search
        await page.locator('//input[@id="searchboxinput"]').fill(f"{service} in {city}")
        await page.keyboard.press("Enter")

        await page.wait_for_selector('//a[contains(@href, "https://www.google.com/maps/place")]', timeout=15000)

        listings = await page.locator('//a[contains(@href, "https://www.google.com/maps/place")]').all()
        listings = listings[:5]

        for i, listing in enumerate(listings):
            try:
                await listing.click()
                await page.wait_for_timeout(3000)  # brief pause after click

                business = Business()

                name_attr = await listing.get_attribute("aria-label")
                business.name = name_attr.strip() if name_attr else ""

                # Address
                addr = page.locator('//button[@data-item-id="address"]//div[contains(@class, "fontBodyMedium")]')
                if await addr.count() > 0:
                    business.address = await addr.first.inner_text()
                else:
                    business.address = city

                # Website
                web = page.locator('//a[@data-item-id="authority"]//div[contains(@class, "fontBodyMedium")]')
                if await web.count() > 0:
                    business.website = await web.first.inner_text()

                # Rating
                rating = page.locator('//div[@jsaction="pane.reviewChart.moreReviews"]//div[@role="img"]')
                if await rating.count() > 0:
                    rating_text = await rating.first.get_attribute("aria-label")
                    if rating_text:
                        business.rating = float(rating_text.split()[0].replace(",", "."))

                # Scroll for reviews
                await page.mouse.wheel(0, 1000)
                await page.wait_for_timeout(2000)
                review_locator = page.locator('//div[@class="MyEned"]//span[@class="wiI7pd"]')
                count = await review_locator.count()
                business.top_reviews = [
                    await review_locator.nth(i).inner_text() for i in range(min(3, count))
                ]

                businesses.append(business)
                # print(f"Collected: {business.name}", flush=True)

                await page.wait_for_timeout(2000)
            except Exception as e:
                # print(f"Error on listing {i+1}: {e}", flush=True)
                continue

        await browser.close()

    return businesses

if __name__ == "__main__":
    import sys
    import asyncio

    if len(sys.argv) != 3:
        print(json.dumps({"error": "Expected query and city as arguments"}))
        sys.exit(1)

    query = sys.argv[1]
    city = sys.argv[2]

    results = asyncio.run(scrape_google_maps(query, city))
    output = [asdict(b) for b in results]

    print(json.dumps(output, ensure_ascii=False), flush=True)
