/*
 * msg_type.h
 *
 *  Created on: Mar 22, 2016
 *      Author: tzhou
 */

#ifndef MSG_TYPE_H_
#define MSG_TYPE_H_

namespace nomad{

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

} // namespace nomad

#endif /* MSG_TYPE_H_ */
