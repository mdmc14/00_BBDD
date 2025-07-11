## Requirements

Before running the script, make sure you meet the following requirements:

- **Python 3.9 or higher**
- **Selenium**
- **Microsoft Edge WebDriver (msedgedriver)** compatible with your Edge browser version
- **`.env` file** with the following variables:
  - `USUARIO`: Your username
  - `PASSWORD`: Your password

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your_user/NAME_REPO.git
   cd NAME_REPO
   
2. **Create the .env file:**
   In the project root, create a .env file with the following variables:
   
   ```bash
   USUARIO=your_username
   PASSWORD=your_password
   
3. **Install dependencies:**
   Ensure you have the necessary dependencies by running the following command:
   
   ```bash
   pip install -r requirements.txt

4. **Install Microsoft Edge WebDriver:**
   If running the script locally, download and install the Microsoft Edge WebDriver. This step is already included in the CI workflow (GitHub Actions).

## Installation
  To run the script locally:
  
  ```bash
  python src/script.py
  ```

The script automates seat reservations for several consecutive days, selecting the specified day and seat.

## GitHub Actions Workflow

The GitHub Actions workflow is configured to run automatically at midnight every day or can be triggered manually. Follow these steps to configure the workflow:
  
1. In your repository, set the secrets `USUARIO` and `PASSWORD` in the "Settings > Secrets and variables > Actions" section on GitHub.

2. The workflow file is located at `.github/workflows/reservar_asiento.yml` and includes the following steps:

  - Download and set up Python 3.9
  - Download and install Microsoft Edge WebDriver
  - Install dependencies
  - Run the reservation script
  - Upload screenshots as artifacts

## Screenshots

During script execution, screenshots are taken to show the status of the reservation. These are saved in the root directory and uploaded as artifacts in GitHub Actions.

## Schedule

The workflow is scheduled to run automatically at midnight every day, with the following cron schedule:

```bash
cron: "0 0 * * *"
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.





   
