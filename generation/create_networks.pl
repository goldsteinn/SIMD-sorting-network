use Algorithm::Networksort;

my $num = $ARGV[0] + 0;
my $algo = $ARGV[1];
my $nw = Algorithm::Networksort->new(inputs => $num, algorithm => $algo);
 
 
print $nw, "\n";
 
