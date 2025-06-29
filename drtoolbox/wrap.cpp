#include <iostream>
#include <octave/oct.h>
#include <octave/octave.h>
#include <octave/parse.h>

int main (int argc, char *argvc[])
{
  // string_vector argv (2);
  // argv(0) = "embedded";
  // argv(1) = "-q";

  // octave_main(2, argv.c_str_vec(), 1);

  // octave_value_list in = octave_value (argvc[1]);
  // octave_value_list out = feval("hellodr.m", in);

  octave_value_list out = feval("hellodr.m");

  if (!error_state && out.length () > 0){}
  else {
    std::cout << "invalid\n";
  }

  return 0;
}
