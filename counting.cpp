#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    int *p;
    int i,j,k;

    for (i=0; i<=10; i++){
        j = i +(i+1)%2;
        k = j >> 1;
        cout << i << '\t' << j << '\t' << k << endl;
    }
   
    for (i=0; i<=10; i++){
        cout << i << endl;
        cout << ~i%2 << endl:
    }
   
}
