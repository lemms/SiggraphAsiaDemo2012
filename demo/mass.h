/*
Siggraph Asia 2012 Demo

Mass vector interface

Laurence Emms
*/

namespace SigAsiaDemo {
	class Mass {
		public:
			Mass(
				float x = 0.0,
				float y = 0.0,
				float z = 0.0,
				float mass = 0.0);
		private:
			float _x;
			float _y;
			float _z;
			float _mass;
	};

	class MassList {
		public:
			MassList();
		private:
			std::vector<Mass> _masses;
	};
}
