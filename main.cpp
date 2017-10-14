#include "utils.h"


extern int equalizeHistRGB(Utils *utils);


int main(int argc, char *argv[])
{
    Utils *utils = new Utils();

    return equalizeHistRGB(utils);
}
