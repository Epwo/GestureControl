import requests
import json


class GestureAPI:
    def __init__(self, ip="192.168.183.216", port=80):
        """
        Initialize the GestureAPI with the given IP and port.

        Args:
            ip (str): The IP address of the API.
            port (int): The port number of the API.
        """
        self.set_api_endpoint(ip, port)
        self.gesture_codes = self.load_gesture_codes()

    def load_gesture_codes(self):
        """Load gesture codes from JSON file."""
        try:
            import os

            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Construct path to the codes.json file
            file_path = os.path.join(script_dir, "../arduino/GestureControl/codes.json")
            # Use absolute path
            with open(os.path.abspath(file_path), "r") as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading gesture codes: {e}")
            # Fallback to default codes if file can't be loaded
            return {
                "up": "0xe41b7f00",
                "down": "0xe41b7f01",
                "left": "0xe41b7f02",
                "right": "0xe41b7f03",
                "ok": "0xe41b7f04",
                "fist": "0xe41b7f05",
            }

    def set_api_endpoint(self, ip, port):
        """
        Set the API endpoint IP and port.

        Args:
            ip (str): The IP address of the API.
            port (int): The port number of the API.
        """
        self.api_ip_port = f"http://{ip}:{port}"
        print(f"API endpoint set to {self.api_ip_port}")

    def send_gesture_code(self, gesture_name):
        """
        Send the hex code corresponding to the gesture name to the API.

        Args:
            gesture_name (str): The name of the gesture.

        Returns:
            dict: The response from the API.
        """
        if gesture_name not in self.gesture_codes:
            return None
        else:
            print(f"Sending gesture '{gesture_name}'")

            try:
                code = self.gesture_codes[gesture_name]
                payload = {"code": code}
            except Exception as e:
                print(f"Error getting gesture code: {e}")
                return None

            try:
                response = requests.post(f"{self.api_ip_port}/send", json=payload)
                response.raise_for_status()  # Raise an exception for HTTP errors
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error sending request: {e}")
                return None


if __name__ == "__main__":
    # Example usage
    api = GestureAPI(ip="localhost", port=5000)

    # Send a gesture
    response = api.send_gesture_code("up")
    if response:
        print(f"Response: {response}")
