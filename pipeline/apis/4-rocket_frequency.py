#!/usr/bin/env python3
"""
Script to fetch and display the number of SpaceX launches per rocket.
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


def get_launches_per_rocket():
    """
    Retrieve and display the number of launches per rocket.

    Returns:
        None
    """
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)

    if response.status_code == 200:
        launches = response.json()

        # Dictionary to count launches per rocket
        rocket_count = {}

        for launch in launches:
            rocket_id = launch['rocket']
            rocket_name = get_rocket_name(rocket_id)
            if rocket_name in rocket_count:
                rocket_count[rocket_name] += 1
            else:
                rocket_count[rocket_name] = 1

        # Sort the results by number of launches and rocket name
        sorted_rockets = sorted(
            rocket_count.items(), key=lambda x: (-x[1], x[0]))

        # Display the results
        for rocket, count in sorted_rockets:
            print(f"{rocket}: {count}")
    else:
        print("Failed to retrieve launch data.")


if __name__ == '__main__':
    get_launches_per_rocket()
