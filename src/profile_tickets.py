"""Command-line entry point for ticket profiling."""
import json
from src.data_pipeline import profile_tickets

if __name__ == "__main__":
    profile = profile_tickets()
    print("Source:", profile["source"])
    print("Tickets profiled:", profile["rows"])
    print("Categories:", profile["categories_count"])
    print("Text length statistics:", profile["text_length_stats"])
    for name, details in profile["categories"].items():
        phrases = ", ".join(f"{x['term']} ({x['count']})" for x in details["keywords"][:8])
        print(f"\n{name}: {details['count']} tickets")
        print("Frequent phrases:", phrases or "No phrase profile available")
    print("\nProfile saved to data/processed/topic_profile.json")
    print("Cleaned text saved to data/processed/tickets_clean.csv")
