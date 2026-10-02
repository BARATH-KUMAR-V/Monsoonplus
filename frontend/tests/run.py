import json, re, sys
from urllib.parse import unquote
from playwright.sync_api import sync_playwright
import fixtures as F

BASE = "http://localhost:4173"
errors, failed_requests = [], []

def install(page):
    def handle(route):
        url = unquote(route.request.url)
        try:
            if "photon.komoot.io" in url or "nominatim" in url:
                body = F.photon()
            elif "router.project-osrm.org" in url:
                body = F.osrm()
            elif "air-quality-api" in url:
                body = F.air()
            elif "/v1/elevation" in url:
                n = len(re.search(r"latitude=([^&]*)", url).group(1).split(","))
                body = F.elevation(n)
            elif "api.open-meteo.com/v1/forecast" in url:
                lats = re.search(r"latitude=([^&]*)", url).group(1).split(",")
                if "past_days" in url:
                    body = F.forecast(days=2)
                elif len(lats) > 1:
                    body = F.forecast_multi(len(lats))
                else:
                    body = F.forecast()
            else:
                return route.continue_()
            route.fulfill(status=200, content_type="application/json", body=json.dumps(body))
        except Exception as exc:  # a broken fixture must be loud, not silent
            failed_requests.append(f"{url} -> fixture error {exc}")
            route.fulfill(status=500, body="{}")
    page.route(re.compile(r"(photon\.komoot|nominatim|project-osrm|open-meteo)"), handle)
    page.route(re.compile(r"tile\.openstreetmap\.org"), lambda r: r.abort())

def shoot(page, name, full=True):
    page.screenshot(path=f"/tmp/shots/{name}.png", full_page=full)

def main(width=430, height=920, tag="m"):
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path="/opt/pw-browsers/chromium")
        page = b.new_page(viewport={"width": width, "height": height})
        page.on("pageerror", lambda e: errors.append(f"[pageerror] {e}"))
        page.on("console", lambda m: errors.append(f"[console.{m.type}] {m.text}")
                if m.type == "error" and "tile" not in m.text.lower() else None)
        install(page)

        page.goto(f"{BASE}/#/")
        page.wait_for_timeout(900)
        # Onboarding must appear on a clean profile.
        assert page.locator("text=What will you use it for?").count() == 1, "onboarding did not show"
        shoot(page, f"{tag}-00-onboard")
        page.click("text=Two-wheeler")
        page.click("text=Continue")
        page.wait_for_timeout(1800)
        shoot(page, f"{tag}-01-home")

        for path, name, wait in [
            ("weather", "02-weather", 1800),
            ("map", "03-map", 2600),
            ("alerts", "04-alerts", 900),
            ("places", "05-places", 900),
            ("report", "06-report", 1200),
            ("ask", "07-ask", 700),
            ("settings", "08-settings", 700),
            ("about", "09-about", 500),
        ]:
            page.goto(f"{BASE}/#/{path}")
            page.wait_for_timeout(wait)
            shoot(page, f"{tag}-{name}")

        # --- the trip flow, end to end ---
        page.goto(f"{BASE}/#/trip")
        page.wait_for_timeout(700)
        page.click("#\\:r0\\: , input[placeholder='Search any place or address']")
        page.fill("input[placeholder='Search any place or address']", "velachery")
        page.wait_for_timeout(900)
        page.locator(".place-opt").nth(1).click()
        page.wait_for_timeout(400)
        page.fill("input[placeholder='Where are you going?']", "thoraipakkam")
        page.wait_for_timeout(900)
        page.locator(".place-opt").nth(3).click()
        page.wait_for_timeout(3000)
        shoot(page, f"{tag}-10-trip")

        has_routes = page.locator(".route-card").count()
        has_why = page.locator("text=How was this worked out?").count()
        if has_why:
            page.locator("text=How was this worked out?").first.click()
            page.wait_for_timeout(400)
            shoot(page, f"{tag}-11-why")

        # Ask, with weather loaded
        page.goto(f"{BASE}/#/ask")
        page.wait_for_timeout(900)
        page.click("text=Will it rain in the next hour?")
        page.wait_for_timeout(500)
        page.fill("#ask-input", "is it safe to ride right now")
        page.click("button:has-text('Ask')")
        page.wait_for_timeout(500)
        shoot(page, f"{tag}-12-ask")

        print(json.dumps({
            "viewport": f"{width}x{height}",
            "routeCards": has_routes,
            "whyPanel": bool(has_why),
            "errors": errors[:25],
            "fixtureFailures": failed_requests[:10],
        }, indent=1))
        b.close()

if __name__ == "__main__":
    main(*( [int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]] if len(sys.argv) > 3 else []))
