import random
import pandas as pd


ticket_templates = {
    "Network": [
        "My Wi-Fi is not connecting",
        "The internet keeps disconnecting",
        "I cannot connect to the company Wi-Fi",
        "My laptop has no internet connection",
        "Wi-Fi keeps dropping on my computer",
        "I am unable to connect to the network",
        "The network connection is very unstable",
        "My internet stopped working",
        "I cannot access the internet from my laptop",
        "The Wi-Fi connection keeps failing"
    ],

    "Access/Account": [
        "I forgot my password",
        "I cannot log into my account",
        "My account is locked",
        "I need access to a shared folder",
        "I need access to a company application",
        "My password is not working",
        "I need permission to access a system",
        "I am unable to sign in",
        "Please reset my account password",
        "I need access to an internal resource"
    ],

    "Hardware": [
        "My laptop will not turn on",
        "The computer is not starting",
        "My laptop screen is not working",
        "The keyboard is not responding",
        "My laptop is overheating",
        "The computer suddenly stopped working",
        "My laptop does not respond when I press the power button",
        "The printer is not working",
        "My computer is making strange noises",
        "The laptop is not charging"
    ],

    "Software": [
        "I need software installed",
        "Please install Microsoft Office",
        "I cannot install the required application",
        "The software installation failed",
        "I need an application installed on my laptop",
        "The application will not open",
        "I need access to approved software",
        "The software is not working after installation",
        "I need a new application for work",
        "The installation keeps failing"
    ],

    "Email": [
        "Outlook is not receiving emails",
        "My emails are not syncing",
        "I cannot send emails",
        "Outlook is disconnected",
        "New emails are not appearing",
        "My Outlook is not working",
        "Emails are not being synchronized",
        "I cannot see my latest emails",
        "Outlook keeps showing offline",
        "My email stopped working"
    ]
}


rows = []

for category, templates in ticket_templates.items():

    for _ in range(100):

        ticket = random.choice(templates)

        # Add small variations to make the dataset less repetitive
        additions = [
            "",
            " Please help.",
            " This started today.",
            " I need this fixed urgently.",
            " It was working yesterday.",
            " I have already restarted my computer.",
            " Can you help me resolve this?",
            " This is preventing me from working."
        ]

        ticket += random.choice(additions)

        rows.append({
            "ticket_text": ticket,
            "category": category
        })


random.shuffle(rows)

df = pd.DataFrame(rows)

df.to_csv("data/tickets.csv", index=False)

print("Dataset created successfully.")
print("Number of tickets:", len(df))
print("\nCategory distribution:")
print(df["category"].value_counts())