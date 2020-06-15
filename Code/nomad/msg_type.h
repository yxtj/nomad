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
	DATA=1,
	CP_START=2,
	CP_FLUSH=3,
	ALIVE=4,
	RESTORE=5,
	DYING=6,
};

} // namespace nomad

#endif /* MSG_TYPE_H_ */
