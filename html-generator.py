import os
import time
import shutil
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

def convert_md2html(md_filename:str='README.md', download_dir:str='/Accounts/turing/students/s24/nguyqu03/Downloads'):
    driver = webdriver.Chrome(executable_path=ChromeDriverManager().install())
    driver.implicitly_wait(0.5)
    driver.maximize_window()
    driver.get("https://codebeautify.org/markdown-to-html");

    # Upload file
    file_input = driver.find_element(by=By.ID, value='fileInput')
    driver.implicitly_wait(5)
    file_input.send_keys(os.path.abspath(md_filename))
    driver.implicitly_wait(5)
    time.sleep(2)
    # Download file
    download_btn = driver.find_element(by=By.XPATH, value='//button[@class="button is-fullwidth "]')
    download_btn.click()
    time.sleep(2)
    # Quit the browser
    driver.quit()

    # Move file to current dir and rename
    filename = 'markdown-to-html.md'    # this is the default filename from the website
    downloaded_path = os.path.join(download_dir, filename)
    saving_path = os.path.dirname(os.path.abspath(md_filename))
    shutil.move(downloaded_path, saving_path)
    os.rename(filename, 'index.html')
    

if __name__ == '__main__':
    convert_md2html()
