import socket, threading, time, sys, argparse
from pythonosc import udp_client, dispatcher, osc_server
import numpy as np
import nibabel

def process_arguments(argv):
	parser = argparse.ArgumentParser(description="NGIMU python-labVIEW")
	parser.add_argument(
		"--ip",
		default="192.168.1.1",
		help="NGIMU IP Address")
	parser.add_argument(
		"--port",
		type=int,
		default=9000,
		help="NGIMU Port")
	parser.add_argument(
		"--receive_port",
		type=int,
		default=8001,
		help="Port to receive messages from NGIMU to this computer")
	parser.add_argument(
		"--udp_ip_address",
		default='192.168.1.2',
		help="UDP IP Address")

	return parser.parse_args(argv[1:])

#https://ubuntuforums.org/showthread.php?t=821325
def get_address():
    try:
        address = socket.gethostbyname(socket.gethostname())
        # On my system, this always gives me 169.254.168.46. Hence...
    except:
        address = ''
    if not address or address.startswith('169.'):
        # ...the hard way.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('4.2.2.1', 0))
        address = s.getsockname()[0]
    return address
    
def main(argv):
    args = process_arguments(argv)
    
    sock_r = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_y = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    sock_r.connect(('127.0.0.1', 777))
    sock_p.connect(('127.0.0.1', 778))
    sock_y.connect(('127.0.0.1', 779))
    
    # Set the NGIMU to send to this machine's IP address
    client = udp_client.SimpleUDPClient(args.ip, args.port)
    client.send_message('/wifi/send/ip', get_address())
    client.send_message('/wifi/send/port', args.receive_port)
	

    def quaternionHandler(add, x, y, z, w):
	    angls = np.array(nibabel.eulerangles.quat2euler([w, x, y, z]))*180/np.pi
	    roll = angls[0]; pitch = -1*angls[1];
	    if(angls[2] < 0): yaw = 180+angls[2]
	    else: yaw = -180+angls[2]
	    print('[roll, pitch, yaw]: [{},{},{}]'.format(roll, pitch, yaw))
	    sock_r.send(str.encode(str(roll)+'\r\n'))
	    sock_p.send(str.encode(str(pitch)+'\r\n'))
	    sock_y.send(str.encode(str(yaw)+'\r\n'))
	    
    dispatch = dispatcher.Dispatcher()
    dispatch.map('/quaternion', quaternionHandler)

    # Set up receiver
    receive_address = args.udp_ip_address, args.receive_port
    server = osc_server.ThreadingOSCUDPServer(receive_address, dispatch)

    print("\nUse ctrl-C to quit.")
    # Start OSCServer
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()

	# Loop while threads are running
    try :
        while 1 :
            time.sleep(10)

    except KeyboardInterrupt :
        server.shutdown()
        server_thread.join()
        return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
