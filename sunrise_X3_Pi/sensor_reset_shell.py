import os,time

# 打开mipi摄像头前重置shell否则会报错
def sensor_reset_shell():
    """
    This function can reset the mipi shell,
    Enable the mipi camera can be used normally.
    """
    os.system("echo 19 > /sys/class/gpio/export")
    os.system("echo out > /sys/class/gpio/gpio19/direction")
    os.system("echo 0 > /sys/class/gpio/gpio19/value")
    time.sleep(0.2)
    os.system("echo 1 > /sys/class/gpio/gpio19/value")
    os.system("echo 19 > /sys/class/gpio/unexport")
    os.system("echo 1 > /sys/class/vps/mipi_host0/param/stop_check_instart")
