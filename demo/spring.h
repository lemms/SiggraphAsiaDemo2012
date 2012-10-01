/*
Siggraph Asia 2012 Demo

Spring vector interface

Laurence Emms
*/

namespace SigAsiaDemo {
	class Spring {
		public:
			Spring(
				unsigned int mass0,
				unsigned int mass1,
				float ks = 0.2,
				float kd = 0.2);
		private:
			unsigned int _mass0; // mass 0 index
			unsigned int _mass1; // mass 1 index
			float _ks; // spring constant
			float _kd; // linear damping
			float _l0;
	};

	class SpringList {
		public:
			SpringList();
		private:
			std::vector<Spring> _springs;
	};
}
