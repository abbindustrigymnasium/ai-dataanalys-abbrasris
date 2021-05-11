from seleniumwire import webdriver
from seleniumwire.request import Request, Response
from selenium.webdriver.chrome.options import Options

from urllib.parse import urlparse
from time import sleep

options = Options()
options.binary_location = "D:/PortableApps/GoogleChromePortable/App/Chrome-bin/chrome.exe"

driver = webdriver.Chrome(options=options)


def response_interceptor(request: Request, response: Response) -> None:
    """
    netloc='ww.google.com'
    path='/recaptcha/api2/payload'
    """
    url = urlparse(request.url)
    print(request)


driver.response_interceptor = response_interceptor

driver.get('http://localhost:5000')

sleep(60)

driver.close()
