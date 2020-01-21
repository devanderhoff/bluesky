from bluesky.network.client import Client
from bluesky.network.server import Server
myclient = Client()


myclient.connect(event_port=11000, stream_port=11001)

myclient.subscribe(b'ACDATA')

def on_event(eventname, eventdata, sender_id):
    print('Event received:', eventname)
    print(eventdata)
    print(sender_id)


def on_stream(streamname, data, sender_id):
    print('Stream data received:', streamname)
    print(data['lat'])
    flag = True
    return flag


flag = False
myclient.event_received.connect(on_event)
flag = myclient.stream_received.connect(on_stream)

myclient.receive()

# print(myclient.servers[myclient.host_id]['nodes'][0])
# diffnode = myclient.servers[myclient.host_id]['nodes'][1]
# myclient.actnode(diffnode)
# myclient.send_event(b'STACKCMD', 'reset')

while not flag:
    myclient.receive()
# while True:
#     myclient.send_event(b'TEST')
#     v = input('Give stack command:')
#     if v:
#         myclient.send_event(b'STACKCMD', v)
#     myclient.receive()
