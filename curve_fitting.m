%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                 Linear regression Matlab version 1.0
%                  created 08/29/2016 by Robert Herrera
%                           Problem 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all;
prompt = 'Please enter file path for curve_fit (problem 1): ';
user_input = input(prompt,'s');

if exist(user_input, 'file')
  % ------------------------------------------------------------------------
 
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Read in file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fstring = fileread(user_input); % read the file as one string

fblocks = regexp(fstring,'[A-Za-z]','split'); % uses any single character as a separator
out = cell(size(fblocks));
for k = 1:numel(fblocks)
    out{k} = textscan(fblocks{k},'%f %f','delimiter',' ','MultipleDelimsAsOne', 1);
    out{k} = horzcat(out{k}{:});
end


a = out{1};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Assign values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = a(1:10,1); % read in x values from text file

t = a(1:10,2); % read in t values from text file

sep = 0.11; %0.11 -> 10 points , 0.01 -> 101 points


for M = 0:9

N = size(x);

phi_x = ones(N(1),M);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        Load polynomial variations into phi(x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:N(1)
  for j = 0:M
      phi_x(i,j+1) = x(i)^j;
  end
end  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       - Find pseudo inverse of phi_x
%       - Find parameter weights based on phi_x_inv * t
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
phi_x_t = pinv(phi_x);

w_user = phi_x_t * t;
disp(w_user)
w_user = fliplr(w_user');

x4 = x;
x_train_size = size(x4);


a = 0;
b = 1;
r = (b-a).*rand(1,10) + a;

x3 = 0:sep:1;%0:sep:1

x_size = size(x3);


x2 = linspace(0,1,100);%0:sep:1
x_test_size = size(x2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    fit training data with parameterss
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y4 = zeros(1,x_train_size(2));
for m = 0:M
 y4 = y4 + w_user(m+1)*x4.^(M-m);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    fit training data with parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y2 = zeros(1,x_test_size(2));
for m = 0:M
 y2 = y2 + w_user(m+1)*x2.^(M-m);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Plot Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x_sin_point = 0:0.01:1;
y_sin = sin(2*pi*x_sin_point);

% Un comment for plots _________________
if M == 0 | M == 1 | M == 3 | M == 9
    figure
    plot(x_sin_point,y_sin,'g',x,t,'bo',x2,y2,'r')
    xlim([-0.1 1.1])
    ylim([-1.5 1.5])
    str = ['M =  ',num2str(M)];
    xlabel('x')
    ylabel('t')
    text(0.8,1,str)
    title('Polynomial Fitting')

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% error mean square both train and test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
e_train = 0.5*(y4 - t).^2; %training


noise_sig = sin(2*pi*x2) + (randn(size(x2))* sqrt(0.30));

e = 0.5*(y2 - noise_sig).^2;  %testing


E_RMS(M+1) =  sqrt((norm(e))/N(1));

E_RMS_Testing(M+1) = sqrt((norm(e_train'))/N(1));

end  



m = 0:9;
figure
plot(m,E_RMS,'-ro',m,E_RMS_Testing,'-bo');
title('Polynomial Fitting :: Noisy Signal');
legend('testing','training')
xlabel('M')
ylabel('E_{RMS}')
xlim([0 9])
ylim([0 1])




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Fitted Noise corrupted sin signal with plots

for N = [15,100]
   
    x_generated = linspace(0,1,N); %original size

    size(x_generated);

    t_generated = sin(2*pi*x_generated);

    noise_signal = randn(size(x_generated)) * sqrt(0.30) + 0.0;

    t_noise = t_generated + noise_signal;

    for M = 9

    n = size(x_generated);

    phi_x = ones(n(2),M);

    for i = 1:n(2)
      for j = 0:M
          phi_x(i,j+1) = x_generated(i)^j;
      end
    end  


    phi_x_t = pinv(phi_x);

    w_user = phi_x_t * t_noise';
    w_user = fliplr(w_user');

    x4 = 0:0.01:1;
    x_train_size = size(x4);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  fit training data with parameters
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    y4 = zeros(1,x_train_size(2));
    for m = 0:M
     y4 = y4 + w_user(m+1)*x4.^(M-m);
    end


    figure;
    plot(x_sin_point,y_sin,'g',x_generated,t_noise,'bo',x4,y4,'r');
    input_string = ['N = ', num2str(n(2))];
    xlabel('x')
    ylabel('t')
    title('Problem 1: Part d');
    text(0.9,1.5,input_string)
    end % end for M
  
end  
  
  
  
  
  
  
  % ------------------------------------------------------------------------
else
  % File does not exist.
  warningMessage = sprintf('Warning: file does not exist:\n%s', user_input);
  uiwait(msgbox(warningMessage));
end
