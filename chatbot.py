import csv
import random

class Chatbot:
    def __init__(self, csv_file):
        self.responses = self.load_responses_from_csv(csv_file)

    def load_responses_from_csv(self, csv_file):
        responses = {}
        with open(csv_file, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                intent = row['intent']
                response = row['response']
                if intent in responses:
                    responses[intent].append(response)
                else:
                    responses[intent] = [response]
        return responses

    def get_response(self, user_input, user_id):
        user_input = user_input.lower()
        if 'menu' in user_input:
            return random.choice(self.responses["menu_inquiry"])
        elif 'hours' in user_input or 'time' in user_input:
            return random.choice(self.responses["hours_inquiry"])
        elif 'reserve' in user_input or 'booking' in user_input:
            return random.choice(self.responses["reservation"])
        elif 'location' in user_input or 'where' in user_input:
            return random.choice(self.responses["location_inquiry"])
        elif 'bill' in user_input or 'invoice' in user_input:
            return random.choice(self.responses["bill_inquiry"])
        elif 'payment' in user_input:
            return random.choice(self.responses["payment_inquiry"])
        elif 'takeaway' in user_input:
            return random.choice(self.responses["takeaway_inquiry"])
        elif 'delivery' in user_input:
            return random.choice(self.responses["delivery_inquiry"])
        elif 'pet' in user_input or 'pets' in user_input:
            return random.choice(self.responses["policy_inquiry"])
        elif 'usama' in user_input or 'parking' in user_input:
            return random.choice(self.responses["facility_inquiry"])
        else:
            return "I'm sorry, I didn't understand that. Can you please rephrase?"
