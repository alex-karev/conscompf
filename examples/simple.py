from conscompf import ConSCompF

# Initialize ConSCompF instance
conscompf = ConSCompF(quiet=True)

# Define data
data: list[dict[str, list[str]]] = [{
        "mistral": [
            "Title: The Unseen Injustice\n\nIn the quiet town of Meadowgrove, a humble schoolteacher named Samuel lived. Known for his dedication to his students and his unwavering honesty, Samuel was well-respected by all. However, beneath this serene facade, a storm of anger was brewing.\n\nOne day, a new principal, Mr. Vanderhoff, was appointed. A man of strict rules and rigid protocols, he was quickly unpopular among the staff. His first act was to scrutinize the school's finances",
            "Title: The Unseen Injustice\n\nIn the bustling city of Metropolis, a humble librarian named Samuel lived a quiet life. He was a man of peace, cherishing the tranquility that books offered. However, beneath his calm exterior simmered a burning anger, a righteous indignation that had been building for years.\n\nEvery day, he would pass by the same corner, where a ruthless landlord named Mr. Abernathy would harass the tenants, raising rents exorbitantly and evicting families without a second thought. Samuel,"
        ],
        "qwen2.5-3b": [
            "In the heart of a bustling city, under the scorching sun, stood a small but sturdy street vendor named Marcus. He had been selling his homemade pastries for years, always with a smile on his face and a warm greeting for each passerby. However, today was different. Today, his bakery’s delivery truck had been delayed by an hour, and he hadn't received the promised ingredients to make his famous chocolate chip cookies. \n\nMarcus watched as customers walked past him without even a glance, their faces occupied with the endless stream of advertisements and billboards that lined the streets. His usual patience was beginning to wane. Finally",
            "In the heart of a bustling city, there stood a small, nondescript apartment in a towering building. It was here that Maria lived, a woman whose life had been turned upside down by a series of unexpected events. Her once-happy marriage had ended in a messy divorce, leaving her with a broken heart and an empty nest. Her job as a marketing executive was also on the line due to budget cuts at her company. And to top it all off, her car had broken down, leaving her stranded for days.\n\nOne evening, Maria received a call from her ex-husband, demanding the return of some of their shared possessions"
        ]
    }, {
        "mistral": [
            "In the heart of a bustling city, where skyscrapers towered like giants and time seemed to race, a curious little girl named Amelia discovered a hidden world beneath the city streets.",
            "In the heart of a bustling city, where the echoes of laughter and the hum of life filled the air, a young girl named Lily found a strange, glowing stone that promised to change her life forever."
        ],
        "qwen2.5-3b": [
            "As the sun set behind the mountains, casting a golden hue over the valley below, she knew this was the moment she had been waiting for.",
            "As the sun began to set on the horizon, casting a warm orange glow over the bustling city, Emily knew it was time to face the daunting task that lay ahead."
        ]
    }
]

# Comprare models
out = conscompf(data, return_type="df")
print(out)

# Output results
print("Similarity matrix:\n\n{}\n\nConsistency scores:\n\n{}".format(
    out["sim_matrix"].round(2).to_markdown(mode="github"),
    out["consistency"].round(4).to_markdown(mode="github")
))
