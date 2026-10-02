"""Regression check: the research view must still work exactly as before."""
import json, re
from urllib.parse import unquote
from playwright.sync_api import sync_playwright
import fixtures as F
from run import install

errors = []
with sync_playwright() as pw:
    b = pw.chromium.launch(executable_path="/opt/pw-browsers/chromium")
    page = b.new_page(viewport={"width": 1280, "height": 1000})
    page.on("pageerror", lambda e: errors.append(f"[pageerror] {e}"))
    page.on("console", lambda m: errors.append(f"[console] {m.text}")
            if m.type == "error" and "ERR_FAILED" not in m.text else None)
    install(page)

    page.goto("http://localhost:4173/#/")
    page.wait_for_timeout(800)
    page.click("text=Car"); page.click("text=Continue")
    page.wait_for_timeout(1200)
    page.goto("http://localhost:4173/#/settings")
    page.wait_for_timeout(600)
    page.click("text=Open developer view")
    page.wait_for_timeout(2000)
    page.screenshot(path="/tmp/shots/dev-00-overview.png", full_page=True)

    found = {}
    for path, name in [("", "overview"), ("trip", "smarttrip"), ("map", "livemap"),
                       ("forecast", "forecast"), ("replay", "replay"),
                       ("model", "modellab"), ("system", "datasystem")]:
        page.goto(f"http://localhost:4173/#/{path}")
        page.wait_for_timeout(1600)
        found[name] = page.locator("h1").count() > 0 and page.locator("text=Page not found").count() == 0
        page.screenshot(path=f"/tmp/shots/dev-{name}.png")

    print(json.dumps({"pagesOk": found, "errors": errors[:15]}, indent=1))
    b.close()
