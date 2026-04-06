from agent.main import app, SearchQuery

# import json
# from langchain_core.load.dump import dumps


async def start_workflow(workflow_id, question, search_type, sio):
    current_step = None  # Track the currently active step

    await sio.emit(
        "workflow_started",
        {"workflow_id": workflow_id},
        room=[workflow_id, "glb_stream"],
    )

    async for event in app.astream_events({"messages": [question], "search_type": search_type}, version="v1"):
        event_type = event["event"]
        name = event.get("name")

        # print("__Runner__", name, event_type)

        if event_type == "on_chain_start":
            current_step = name  # Update current step when a chain starts
            await sio.emit(
                "step_started",
                {"workflow_id": workflow_id, "step": name},
                room=[workflow_id, "glb_stream"],
            )

        # Add handler for on_chat_model_start if you want to track when the model begins
        # if event_type == "on_chat_model_start":
        #     print(f"Chat model started for step: {current_step}")
            # You can emit an event here if needed, e.g.:
            # await sio.emit(
            #     "chat_model_started",
            #     {"workflow_id": workflow_id, "step": current_step},
            #     room=[workflow_id, "glb_stream"],
            # )

        if event_type == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            print(token)
            # Now you know this stream is from the current_step
            # print(f"Streaming token for step: {current_step}")
            await sio.emit(
                "token_stream",
                {"token": token, "step": current_step},  # Include step in the emit
                room=[workflow_id, "glb_stream"],
            )

        if event_type == "on_chain_end":
            # print("on_chain_end", name)

            if name == "route_to_scrape":
                output = None
            else:
                output = event["data"]["output"]
                if isinstance(output, SearchQuery):
                    output = output.model_dump()

            await sio.emit(
                "step_completed",
                {
                    "workflow_id": workflow_id,
                    "step": name,
                    "output": output,
                },
                room=[workflow_id, "glb_stream"],
            )

            # Optionally reset current_step if the chain ends
            if name == current_step:
                current_step = None

    await sio.emit(
        "workflow_finished",
        {"workflow_id": workflow_id},
        room=[workflow_id, "glb_stream"],
    )
