import uuid
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
import socketio

from server.sockets import sio
from agent.runner import start_workflow

app = FastAPI()

socket_app = socketio.ASGIApp(sio, other_asgi_app=app)


class WorkflowRequest(BaseModel):
    question: str


@app.post("/workflow/start")
async def start(req: WorkflowRequest):

    workflow_id = str(uuid.uuid4())

    # start workflow after small delay
    asyncio.create_task(delayed_start(workflow_id, req.question))

    return {"workflow_id": workflow_id}


async def delayed_start(workflow_id, question):

    await asyncio.sleep(2)

    await start_workflow(workflow_id, question, sio)


@sio.event
async def connect(sid, environ):
    print("connected", sid)


@sio.event
async def disconect(sid, environ):
    print("Disconnected", sid)


@sio.event
async def join_room(sid, data):
    print("join", data)
    workflow_id = data["workflow_id"]
    print(workflow_id)

    await sio.enter_room(sid, workflow_id)

    await sio.emit("room_joined", {"workflow_id": workflow_id}, room=sid)
