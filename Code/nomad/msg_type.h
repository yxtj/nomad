#ifndef MSG_TYPE_H_
#define MSG_TYPE_H_

enum MsgType{
	DATA = 1,
	CP_START = 2,
	CP_CLEAR = 3,
	CP_LFINISH = 4,
	CP_RESUME = 5,
	CP_RESTORE = 6,
	LOCAL_ERROR = 7,
	TERMINATION = 10,
};

#endif /* MSG_TYPE_H_ */
