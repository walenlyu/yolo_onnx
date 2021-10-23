#include<string>
using namespace std;

struct Net_config
{
	float confThreshold; // class Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	float objThreshold;  //Object Confidence threshold
	string netname;
};