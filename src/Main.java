import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.Arrays;

class Constant {

  public static final Point START = new Point(3, 0);
  public static final Point GOAL = new Point(3, 7);
  public static final Action ACTION_LIST[] = new Action[]{Action.UP,
      Action.DOWN,
      Action.LEFT,
      Action.RIGHT,
      Action.RIGHT_UP,
      Action.RIGHT_DOWN,
      Action.LEFT_UP,
      Action.LEFT_DOWN,
      Action.NO_CHANGE};

  public static final int WIND[] = new int[]{0, 0, 0, 1, 1, 1, 2, 2, 1, 0};
  public static final double EPSILON = 0.1;
  public static final double LEARNING_RATE = 0.5;
  public static final double REWARD = -1.0;
  public static final int WORLD_HEIGHT = 7;
  public static final int WORLD_WIDTH = 10;
  public static final int EPISODE_LIMIT = 100;
  public static final double DISCOUNT_RATE = 0.9;

  @Override
  public String toString() {
    return "Constant{}";
  }
}

enum Action {
  UP(-1, 0, "U"),
  DOWN(1, 0, "D"),
  LEFT(0, -1, "L"),
  RIGHT(0, 1, "R"),
  RIGHT_UP(-1, 1, "RU"),
  RIGHT_DOWN(1, 1, "RD"),
  LEFT_UP(-1, -1, "LU"),
  LEFT_DOWN(1, -1, "LD"),
  NO_CHANGE(0, 0, "-");

  private final int xChange;
  private final int yChange;
  private String name;

  public String getName() {
    return name;
  }

  public int getxChange() {
    return xChange;
  }

  public int getyChange() {
    return yChange;
  }

  public static Action getAction(int ordinal) {
    for (Action action : Action.values()) {
      if (action.ordinal() == ordinal) {
        return action;
      }
    }
    System.out.println("Not found:" + ordinal);
    return null;
  }

  Action(int xChange, int yChange, String name) {
    this.xChange = xChange;
    this.yChange = yChange;
    this.name = name;
  }
}

class Point implements Comparable<Point> {

  int x;
  int y;

  public Point(int x, int y) {
    this.x = x;
    this.y = y;
  }

  public String toString() {
    return "(" + x + "," + y + ")";
  }

  @Override
  public int compareTo(Point p) {
    if (p.x == this.x && p.y == this.y) {
      return 1;
    }
    return 0;
  }
}

class Grid {

  private Point start;
  private Point goal;
  private int height;
  private int width;
  private double qValue[][][];

  public Point getStart() {
    return start;
  }

  public Point getGoal() {
    return goal;
  }

  public int getHeight() {
    return height;
  }

  public int getWidth() {
    return width;
  }

  public Grid(Point start, Point goal, int height, int width, int noOfPossibleAction) {
    this.start = start;
    this.goal = goal;
    this.height = height;
    this.width = width;
    this.qValue = new double[this.height][this.width][noOfPossibleAction];
  }

  public double getQValue(int i, int j, Action action) {
    return this.qValue[i][j][action.ordinal()];
  }

  public double[] getQValues(int i, int j) {
    return this.qValue[i][j];
  }

  public void setQValue(int i, int j, Action action, double qVal) {
    this.qValue[i][j][action.ordinal()] = qVal;
  }

  public void printGrid(int episodeNumber) {
    System.out.println("Result Of Episode -> " + episodeNumber);
    for (int i = 0; i < height; i++) {
      System.out.print("|");
      for (int j = 0; j < width; j++) {
        System.out.print(Action.getAction(Utility.getMaxIndex(getQValues(i, j))).getName() + " |");
      }
      System.out.println();
    }
  }
}

class GridWorld {

  private Grid grid;

  private Action[] actionList;
  private int[] wind;

  public Grid getGrid() {
    return grid;
  }

  public GridWorld(Point start, Point goal, int gridHeight, int gridWidth, Action[] actionList,
      int[] wind) {
    this.actionList = actionList;
    this.wind = wind;
    this.grid = new Grid(start, goal, gridHeight, gridWidth, actionList.length);
  }

  public Point getNextStep(Point currentState, Action action) {
    int i = currentState.x;
    int j = currentState.y;

    int newI = i + action.getxChange() - this.wind[j];
    int newJ = j + action.getyChange();

    if (newI < 0) {
      newI = 0;
    }
    if (newJ < 0) {
      newJ = 0;
    }
    if (newI >= this.grid.getHeight()) {
      newI = this.grid.getHeight() - 1;
    }
    if (newJ >= this.grid.getWidth()) {
      newJ = this.grid.getWidth() - 1;
    }
    return new Point(newI, newJ);
  }

  public Action getRandomAction() {
    return Action.getAction((int) (Math.random() * actionList.length));
  }

}

class WindyGreedWorld {

  GridWorld gridWorld;
  private double reward;
  private double alpha;
  private double epsilon;
  private int episodeLimit;
  private double discountRate;

  public void initialization(int episodeLimit, Point start, Point goal, int gridHeight,
      int gridWidth, Action[] actionList, int[] wind, double reward, double alpha, double epsilon,
      double discountRate) {
    this.episodeLimit = episodeLimit;
    this.reward = reward;
    this.alpha = alpha;
    this.epsilon = epsilon;
    this.discountRate = discountRate;
    gridWorld = new GridWorld(start, goal, gridHeight, gridWidth, actionList, wind);
  }

  public int episode(boolean isDebug) {
    int time = 0;
    Point state = gridWorld.getGrid().getStart();
    Action action;
    Point nextState;
    Action nextAction;
    if (Utility.getBinomial(1, this.epsilon) == 1) {
      action = gridWorld.getRandomAction();
    } else {
      double values[] = gridWorld.getGrid().getQValues(state.x, state.y);
      action = Action.getAction(Utility.getMaxIndex(values));
    }
    while (state.compareTo(gridWorld.getGrid().getGoal()) != 1) {

      nextState = gridWorld.getNextStep(state, action);
      System.out
          .println("----------------------------------------------------------------------------");
      System.out.println(
          "Current State is " + state + "  Action : " + action.getName() + " Next State is "
              + nextState);
      if (Utility.getBinomial(1, this.epsilon) == 1) {
        nextAction = gridWorld.getRandomAction();
      } else {
        double values[] = gridWorld.getGrid().getQValues(nextState.x, nextState.y);
        nextAction = Action.getAction(Utility.getMaxIndex(values));
      }
      if (isDebug) {
        System.out.println(nextState.toString() + " Action:" + action.name());
      }

      double sarsaValue = calculateSarsa(action,
          nextAction, state, nextState);
      double updatedQvalue =
          gridWorld.getGrid().getQValue(state.x, state.y, action) + sarsaValue;
      System.out.println(
          "Q value of state " + state + "  " + gridWorld.getGrid()
              .getQValue(state.x, state.y, action));
      System.out.println(
          "Updated Q Value of Current State (previous Q value plus SARSA value)" + state + " "
              + gridWorld.getGrid().getQValue(state.x, state.y, action) + " + " + sarsaValue + " = "
              + updatedQvalue);
      gridWorld.getGrid().setQValue(state.x, state.y, action, updatedQvalue);
      System.out
          .println("-----------------------------------------------------------------------------");
      state = nextState;
      action = nextAction;
      time += 1;
    }
    return time;
  }

  public double calculateSarsa(Action action, Action nextAction, Point state, Point nextState) {
    double calucatedSarsa = alpha * (reward + discountRate * (gridWorld.getGrid()
        .getQValue(nextState.x, nextState.y, nextAction)) -
        gridWorld.getGrid().getQValue(state.x, state.y, action));
    System.out.println(
        "calculated Sarsa " + calucatedSarsa);
    return calucatedSarsa;
  }

  public void run() {
    System.out.println("Episode Limit" + Constant.EPISODE_LIMIT);
    System.out.println("Start Point " + Constant.START);
    System.out.println("Destination Point " + Constant.GOAL);
    System.out.println("World Height " + Constant.WORLD_HEIGHT);
    System.out.println("Action List " + Arrays.toString(Constant.ACTION_LIST));
    System.out.println("Wind " + Arrays.toString(Constant.WIND));
    System.out.println("Reward " + Constant.REWARD);
    System.out.println("Alpha " + Constant.LEARNING_RATE);
    System.out.println("Epsilon " + Constant.EPSILON);
    System.out.println("Discount Rate " + Constant.DISCOUNT_RATE);

    initialization(Constant.EPISODE_LIMIT, Constant.START, Constant.GOAL, Constant.WORLD_HEIGHT,
        Constant.WORLD_WIDTH, Constant.ACTION_LIST, Constant.WIND, Constant.REWARD,
        Constant.LEARNING_RATE,
        Constant.EPSILON, Constant.DISCOUNT_RATE);
    int episodeNum = 0;
    int time = 0;
    while (episodeNum++ < episodeLimit) {
      boolean isDebug = true;
      if (episodeNum >= episodeLimit - 3) {
        isDebug = true;
      }
      int episodeTime = episode(isDebug);
      System.out.println("Episode Time:" + episodeTime);
      time += episodeTime;
      gridWorld.getGrid().printGrid(episodeNum);
    }
    System.out.println("Total Time:" + time);

  }
}

public class Main {

  public static void main(String[] args) {

    try {
      PrintStream o = new PrintStream(new File("output.txt"));
      System.setOut(o);
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
    new WindyGreedWorld().run();
  }
}

class Utility {

  public static int getBinomial(int n, double p) {
    int x = 0;
    for (int i = 0; i < n; i++) {
      if (Math.random() < p) {
        x++;
      }
    }
    return x;
  }

  public static int getMaxIndex(double[] values) {
    int maxIndex = 0;
    for (int i = 0; i < values.length; i++) {
      if (values[i] > values[maxIndex]) {
        maxIndex = i;
      }
    }
    return maxIndex;
  }
}