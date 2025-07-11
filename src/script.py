# Load libraries and environment variables
import os
import time
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from datetime import datetime
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# Cargar las variables de entorno desde el archivo .env
if load_dotenv():
    print("Variables de entorno cargadas.")
else:
    print("No se pudieron cargar las variables de entorno.")

# Acceder a las variables de entorno
USUARIO = os.getenv('USUARIO')
CONTRASEÑA = os.getenv('PASSWORD')

# Función para calcular la posición del día en el calendario (41 posiciones)
def calcular_posicion_calendario(dia, mes, anio):
    dias_por_mes = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Ajuste para años bisiestos
    if anio % 4 == 0 and (anio % 100 != 0 or anio % 400 == 0):
        dias_por_mes[1] = 29

    # Calcular el día de la semana del 1 del mes
    a = (14 - mes) // 12
    y = anio - a
    m = mes + 12 * a - 2
    dia_semana_1_mes = (1 + y + y // 4 - y // 100 + y // 400 + (31 * m) // 12) % 7

    # Ajustar para que el domingo sea 6 en lugar de 0
    dia_semana_1_mes = 6 if dia_semana_1_mes == 0 else dia_semana_1_mes - 1

    # Calcular la posición en el calendario de 41 posiciones
    posicion = dia_semana_1_mes + dia - 1

    # Ajustar si el día es del mes anterior
    if posicion < 0:
        mes_anterior = 12 if mes == 1 else mes - 1
        dias_mes_anterior = dias_por_mes[mes_anterior - 1]
        posicion = dias_mes_anterior + posicion

    return posicion

# Función para hacer la reserva
def hacer_reserva(USUARIO, CONTRASEÑA, dias, asiento, oficinas):
            # Iterar sobre los días proporcionados para hacer la reserva
    for dia in dias:
        try:
            # Configurar opciones para ejecutar Edge en modo headless
            edge_options = EdgeOptions()
            edge_options.use_chromium = True
            edge_options.add_argument('--headless')  # Modo headless
            # edge_options.add_argument("window-size=1920x1080")
            # edge_options.add_argument('--disable-gpu')  # Deshabilitar GPU (opcional pero recomendado)
            # edge_options.add_argument('--no-sandbox')  # Evitar problemas de permisos
            # edge_options.add_argument('--disable-dev-shm-usage')  # Para evitar problemas de memoria compartida
            
            # Configurar el servicio con la ruta al ejecutable de msedgedriver
            edge_service = Service('/usr/local/bin/msedgedriver')  # Ruta para GitHub Actions
            
            # Inicializar el navegador usando el servicio y opciones
            driver = webdriver.Edge(service=edge_service, options=edge_options)
            
            # Navegar al sitio de reservas
            driver.get("https://allianz.swa.steelcase.com/")
            time.sleep(5)
    
            # Introducir el usuario y la contraseña
            driver.find_element(By.ID, "username").send_keys(USUARIO)
            driver.find_element(By.ID, "password").send_keys(CONTRASEÑA)
    
            # Hacer clic en el botón de iniciar sesión
            boton_iniciar_sesion = driver.find_element(By.CSS_SELECTOR, 'button[data-action-button-primary="true"][value="default"]')
            boton_iniciar_sesion.click()
            time.sleep(10)
    
    
            # Hacer clic en "Solicitar reserva de puesto de trabajo"
            driver.find_element(By.LINK_TEXT, "Solicitar reserva de puesto de trabajo").click()
            time.sleep(10)
            
            # Hacer clic en el botón para elegir la fecha
            driver.find_element(By.CSS_SELECTOR, ".title > .btn:nth-child(1)").click()
            time.sleep(3)
    
            # Calcular la posición del día en el calendario
            fecha_actual = datetime.now()
            mes_actual = fecha_actual.month
            anio_actual = fecha_actual.year
            posicion = calcular_posicion_calendario(dia, mes_actual, anio_actual)
    
            # Seleccionar el día en el calendario
            driver.find_element(By.XPATH, f"//*[starts-with(@id, 'datepicker-') and contains(@id, '-{posicion}')]/button").click()
    
            # Confirmar la reserva
            driver.find_element(By.XPATH, "/html/body/div[1]/div/div/div[3]/button[2]").click()

            # Seleccionar la hora de inicio de la reserva (Desde las 7 am)
            driver.find_element(By.ID, "Desde").click()
            for _ in range(1):
                driver.find_element(By.XPATH, "//*[@id='modal-body']/div/table/tbody/tr[1]/td[1]/a").click()
            time.sleep(10)
 
            # Confirmar la selección
            driver.find_element(By.XPATH, "/html/body/div[1]/div/div/div[3]/button[2]").click()
            time.sleep(10)
            
            # Seleccionar la hora de fin de la reserva (por defecto hasta las 17:00)
            driver.find_element(By.ID, "Hasta").click()
            for _ in range(10):
                driver.find_element(By.XPATH, "//*[@id='modal-body']/div/table/tbody/tr[1]/td[1]/a").click()
    
            time.sleep(10)
            
            # Confirmar la selección
            driver.find_element(By.XPATH, "/html/body/div[1]/div/div/div[3]/button[2]").click()
            time.sleep(10)
            driver.save_screenshot("reserva_asiento0.png")

            # Abre una nueva ventana para realizar la reserva según la oficina/planta
            driver.find_element(By.XPATH, f"/html/body/div[1]/div[2]/div[2]/div/div[2]/div[2]/div[1]/div[{oficinas}]").click()
            
            time.sleep(30)
            driver.save_screenshot("reserva_asiento.png")
            try:
                driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/div/div[2]/div/div/div[2]/div/div/div[3]/div[1]/div/a[2]").click()
            except: # Más tiempo de espera por si acaso
                time.sleep(30)
                driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/div/div[2]/div/div/div[2]/div/div/div[3]/div[1]/div/a[2]").click()
            #driver.save_screenshot("reserva_asiento3.png")
            time.sleep(10)
            driver.save_screenshot("reserva_asiento4.png")
            #driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/div/div[2]/div/div/div[2]/div/div/div[2]/div[4]/img[24]").click()
            # Usar una f-string para insertar la variable 'asiento' en el XPath
            driver.find_element(By.XPATH, f"/html/body/div[1]/div[2]/div[2]/div/div[2]/div/div/div[2]/div/div/div[2]/div[4]/img[{asiento}]").click()
    
            time.sleep(30)
            driver.save_screenshot("reserva_asiento5.png")
            # Seleccionar el asiento especificado
            # driver.get(f"https://allianz.swa.steelcase.com/reservas/#!/realizar-reserva/{asiento}")
            try:
                
                driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/div/div[2]/div/div[2]/div[2]/ul[2]/li[3]/button").click()

            except:
                time.sleep(30)
                driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/div/div[2]/div/div[2]/div[2]/ul[2]/li[3]/button").click()
                
            driver.save_screenshot("reserva_asiento6.png")
            time.sleep(15)
            # Confirmar la reserva final
            driver.find_element(By.XPATH, "/html/body/div[1]/div/div/div[3]/button[2]").click()
            
    
            print(f"Reserva realizada para el día {dia} con éxito en el asiento {asiento}.")
    
            # Esperar un momento antes de realizar la siguiente reserva (si es necesario)
            time.sleep(5)
    
            # Volver a la página principal
            driver.get("https://allianz.swa.steelcase.com/")
            time.sleep(15)
            
            # Cerrar el navegador al terminar
            driver.quit()

        except Exception as e:
            print(f"Error durante la reserva: {str(e)}")

# Ejecutar el script
if __name__ == "__main__":
    fecha = datetime.now()
    dia = int(fecha.day)
    dias = [dia + i for i in range(2, 6)]  
    asiento = 16  
    oficinas = 4
    hacer_reserva(USUARIO, CONTRASEÑA, dias, asiento, oficinas)
