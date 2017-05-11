x = -10:0.01:10;
y1 = sigmoid(x);
y2 = tanh(x);
y3 = x
y3(y3<=0) = 0
figure(1)
plot(x,y1,'k',...
       'LineWidth',5,...
       'MarkerSize',5,...
       'MarkerEdgeColor','b',...
       'MarkerFaceColor',[0.5,0.5,0.5])
xlabel('x','FontName','Times New Roman','FontSize',24);
ylabel('y','FontName','Times New Roman','FontSize',24);
set(gca,'FontName','Times New Roman','FontSize',24);
set(gcf,'Position',[1,1,1500,1000]);
grid on

% figure(2)
hold on
plot(x,y2,'r',...
      'LineWidth',5,...
       'MarkerSize',5,...
       'MarkerEdgeColor','b',...
       'MarkerFaceColor',[0.5,0.5,0.5])
xlabel('x','FontName','Times New Roman','FontSize',24);
ylabel('y','FontName','Times New Roman','FontSize',24);
set(gca,'FontName','Times New Roman','FontSize',24);
set(gcf,'Position',[1,1,1500,1000]);
 grid on
 
 
% figure(3)
hold on
plot(x,y3,'b',...
      'LineWidth',5,...
       'MarkerSize',5,...
       'MarkerEdgeColor','b',...
       'MarkerFaceColor',[0.5,0.5,0.5])
xlabel('x','FontName','Times New Roman','FontSize',24);
ylabel('y','FontName','Times New Roman','FontSize',24);
set(gca,'FontName','Times New Roman','FontSize',24);
set(gcf,'Position',[1,1,1500,1000]);
grid on
legend('sigmoid','tanh','relu')
   







