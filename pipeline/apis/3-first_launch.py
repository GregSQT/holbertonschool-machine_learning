#!/usr/bin/env python3
"""
Uses the (unofficial) SpaceX API to print the upcoming launch
"""

import requests


def get_rocket_name(rocket_id):
    """
    Retrieve the name of the rocket using its ID.

    Args:
        rocket_id (str): The ID of the rocket.

    Returns:
        str: The name of the rocket.
    """
    rocket_url = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    response = requests.get(rocket_url)
    if response.status_code == 200:
        rocket_data = response.json()
        return rocket_data.get('name', 'Unknown')
    return 'Unknown'


def get_first_launch():
    """
    Retrieve and display information about the first SpaceX launch.

    Returns:
        str: A formatted string with the launch details.
    """
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    response = requests.get(url)

    if response.status_code == 200:
        launches = response.json()

        # Sort launches by date_unix
        launches_sorted = sorted(launches, key=lambda x: x['date_unix'])

        # Get the first launch in the sorted list
        first_launch = launches_sorted[0]

        # Extract necessary information
        launch_name = first_launch.get('name', 'Unknown')
        date = first_launch.get('date_local', 'Unknown')
        rocket_id = first_launch.get('rocket', 'Unknown')
        launchpad_id = first_launch.get('launchpad', 'Unknown')

        # Fetch rocket name using the rocket ID
        rocket_name = get_rocket_name(rocket_id)

        # Fetch launchpad details using the launchpad ID
        launchpad_url = (
            f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}")
        launchpad_response = requests.get(launchpad_url)
        if launchpad_response.status_code == 200:
            launchpad_data = launchpad_response.json()
            launchpad_name = launchpad_data.get('name', 'Unknown')
            launchpad_locality = launchpad_data.get('locality', 'Unknown')
        else:
            launchpad_name = 'Unknown'
            launchpad_locality = 'Unknown'

        # Format the output
        return (f"{launch_name} ({date}) {rocket_name} - "
                f"{launchpad_name} ({launchpad_locality})")
    else:
        return "Failed to retrieve launch data."


if __name__ == '__main__':
    launch_info = get_first_launch()
    print(launch_info)
