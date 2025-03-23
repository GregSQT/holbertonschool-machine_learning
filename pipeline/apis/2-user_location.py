#!/usr/bin/env python3
"""
Uses the GitHub API to print the location of a specific user,
where user is passed as first argument of the script with full API URL
"""

import requests
import sys
from datetime import datetime


def get_user_location(url):
    """
    Retrieve the location of a GitHub user from the API.

    Args:
        url (str): The API URL for the GitHub user.

    Returns:
        str: The location of the user or an error message.
    """
    response = requests.get(url)

    if response.status_code == 404:
        return "Not found"
    elif response.status_code == 403:
        # Handle rate limiting
        reset_time = response.headers.get('X-Ratelimit-Reset')
        if reset_time:
            reset_time = int(reset_time)
            current_time = datetime.utcnow().timestamp()
            time_difference = reset_time - current_time
            minutes_to_reset = divmod(time_difference, 60)[0]
            return f"Reset in {int(minutes_to_reset)} min"
        return "Rate limit exceeded"
    elif response.status_code == 200:
        data = response.json()
        location = data.get('location', 'Location not set')
        return location
    else:
        return f"Unexpected status code: {response.status_code}"


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API URL>")
        sys.exit(1)

    url = sys.argv[1]
    location = get_user_location(url)
    print(location)
