# This message contains essential time information of the 
# local sensor (visual/inertial sensor) when get triggered
std_msgs/Header header   # local timestamp of the trigger
uint32                 trigger_id
uint32                 event_id
# time                   timestamp_host       # corresponding timestamp of the Host PC, not used in GVINS