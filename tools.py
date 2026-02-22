from crewai_tools import YoutubeChannelSearchTool

yt_tool = YoutubeChannelSearchTool(
    youtube_channel_handle="https://www.youtube.com/channel/UCjWY5hREA6FFYrthD0rZNIw",
    config={
        "embedder": {
            "provider": "ollama",   
            "model": "gemma:2b"
        },
        "llm": {
            "provider": "ollama",  
            "model": "gemma:2b"
        },
        "use_openai": False         
    }
)
