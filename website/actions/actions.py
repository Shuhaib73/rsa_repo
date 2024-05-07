# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from datetime import datetime

class ActionGetCurrentDate(Action):

    def name(self) -> Text:
        return "action_get_current_date"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Extract user's requested information
        requested_info = tracker.latest_message['entities'][0]['entity']

        # Get current date
        current_date = datetime.now()

        if requested_info in ['year', 'Year']:
            response = f"The current year is {current_date.strftime('%Y')}"
        elif requested_info in ['month', 'Month']:
            response = f"The current month is {current_date.strftime('%B')}"
        elif requested_info in ['day', 'Day']:
            response = f"The current day is {current_date.strftime('%A')}"
        else:
            response = f"The current date is {current_date.strftime('%Y-%m-%d')}"

        # Send the response back to the user
        dispatcher.utter_message(response)

        return []


class ActionCalculateAge(Action):

    def name(self) -> Text:
        return "action_calculate_age"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Get user's date of birth
        dob_entity = next(tracker.get_latest_entity_values("dob"), None)  # Extract DOB entity from tracker

        if dob_entity:
            dob_datetime = datetime.strptime(dob_entity, '%Y-%m-%d')
            current_datetime = datetime.now()

            age = current_datetime.year - dob_datetime.year - ((current_datetime.month, current_datetime.day) < (dob_datetime.month, dob_datetime.day))

            response = f"You are {age} years old."
        else:
            response = "Sorry, I couldn't understand your date of birth."


        dispatcher.utter_message(response)

        return []