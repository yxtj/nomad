#ifndef MSG_TYPE_H_
#define MSG_TYPE_H_

enum MsgType : int8_t{
	DATA = 0,
	TERMINATION = 1,
	LOCAL_ERROR = 2,
	CP_START = 11,
	CP_CLEAR = 12,
	CP_LFINISH = 13,
	CP_RESUME = 14,
	CP_RESTORE = 15,
};

#endif /* MSG_TYPE_H_ */
