#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
   int *p;
   int i,j,k;

   cout << "Round up to nearest odd number" << endl;  
   for(i=0; i<10; i++){
        j = i + (i+1)%2;
        k = j >> 1;
        cout << i << '\t' << j << '\t' << k << endl;
    }

    for(i=0; i<10; i++){
        cout << i << '\t' << ~i%2 <<endl;
    }

    // Testing if-continue statement
    for (i=0; i<10; i++){   
        for (j=0; j<10; j++){
            //cout << i << '\t' << j << endl;
            if(i==5) continue; // If condition is met continue to next iteration.
            cout << i << '\t' << j << endl;       
        }
    }
   
}
