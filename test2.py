from bluesky.network.client import Client

myclient = Client()
myclient.connect(event_port=11000, stream_port=11001)
# myclient.subscribe(b'ACDATA')

staterecvd = False

def on_event(eventname, eventdata, sender_id):
    if eventname == b'MLSTATEREPLY':
        print('ML State reply received:', eventname)
        global staterecvd
        staterecvd = True

myclient.event_received.connect(on_event)
# On startup, wait for server information
myclient.receive(1000)

while True:
    staterecvd = False
    v = input('Press enter for next simulation step:')
    myclient.send_event(b'STACKCMD', 'MLSTEP')
    while not staterecvd:
        myclient.receive(1)
