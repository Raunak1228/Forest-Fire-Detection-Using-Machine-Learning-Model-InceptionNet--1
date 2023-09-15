from twilio.rest import Client

# Twilio credentials
account_sid = 'AC8a43e1b89274f9d62be3ed72a01812ab'
auth_token = '486cd4d26f1aa294f41000a6786f00a6'

# Create a Twilio client
client = Client(account_sid, auth_token)

# Phone numbers
sender_phone_number = '+12707516892'  # Your Twilio phone number
recipient_phone_number = '+918805163155'  # The recipient's phone number

# Message content
message_body = 'Fire detected! Please take necessary action.'

# Send the SMS message
message = client.messages.create(
    body=message_body,
    from_=sender_phone_number,
    to=recipient_phone_number
)

# Print the message SID
print('Message sent. SID:', message.sid)
