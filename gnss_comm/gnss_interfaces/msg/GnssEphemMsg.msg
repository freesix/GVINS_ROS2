# This message contains GPS, Galileo and BeiDou ephemeris data

uint32 sat  # 卫星id(在gnss_utility.hpp文件的'sat_no'函数中定义)
GnssTimeMsg ttr # GPST中的传输时间
GnssTimeMsg toe # GPST中的星历参考时间(GPS周内秒)
GnssTimeMsg toc # GPST中的时钟矫正参考时间
float64 toe_tow
uint32 week
uint32 iode # 数据、星历发布时间(数据期龄)
uint32 iodc # 钟的数据龄期
uint32 health
uint32 code
float64 ura
float64 a
float64 e # 轨道偏心率
float64 i0 # toe时的轨道倾角
float64 omg # 轨道近地角距
float64 omg0 # 周内时等于0时的轨道升交点赤经
float64 m0 # 参考时间toe的平近点角(rad)
float64 delta_n # 平均运动角速度矫正值
float64 omg_dot # 轨道升交点赤经对时间的变化率
float64 i_dot # 参考时间的轨道倾角
float64 cuc # 维度幅角的余弦调和改正项的振幅(rad)
float64 cus # 轨道幅角的正弦调和改正项的振幅(rad)
float64 crc # 轨道平径的余弦调和改正项的振幅(rad)
float64 crs # 轨道半径的正弦调和改正项的振幅(m)
float64 cic # 轨道倾角的余弦调和改正项的振幅(rad)
float64 cis # 维度倾角的正弦调和改正项(rad)
float64 af0 # 卫星时钟矫正参数
float64 af1 # 卫星时钟矫正参数
float64 af2 # 卫星时钟矫正参数
float64 tgd0
float64 tgd1
float64 a_dot
float64 n_dot