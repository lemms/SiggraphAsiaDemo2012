/*
Siggraph Asia 2012 Demo

Creator interface.

Laurence Emms
*/

namespace SigAsiaDemo {
	class MassList;
	class SpringList;
	class Creator {
		public:
			virtual void create(
				float x,	// position
				float y,
				float z,
				MassList &masses,
				SpringList &springs) = 0;
	};
}
