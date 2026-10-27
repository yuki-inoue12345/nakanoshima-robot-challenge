#include "gtest/gtest.h"
#include "robotbase_bt_nodes/passage.hpp"

using robotbase_bt_nodes::traversed;

TEST(Passage, DoesNotRemoveOneMetreBeforeGoal)
{
  EXPECT_FALSE(traversed({0, 0}, {2, 0}, {4, 0}, {1, 0}));
  EXPECT_FALSE(traversed({0, 0}, {2, 0}, {4, 0}, {1.99, 0}));
  EXPECT_TRUE(traversed({0, 0}, {2, 0}, {4, 0}, {2.01, 0}));
}
TEST(Passage, HandlesSmoothCornerWithoutAnArrivalRadius)
{
  EXPECT_FALSE(traversed({0, 0}, {2, 0}, {2, 2}, {1, 0}));
  EXPECT_TRUE(traversed({0, 0}, {2, 0}, {2, 2}, {1.9, .3}));
}
TEST(Passage, HandlesReverseDirectionAndInitialCoincidentPoint)
{
  EXPECT_FALSE(traversed({4, 0}, {2, 0}, {0, 0}, {3, 0}));
  EXPECT_TRUE(traversed({4, 0}, {2, 0}, {0, 0}, {1.9, 0}));
  EXPECT_TRUE(traversed({0, 0}, {0, 0}, {2, 0}, {0, 0}));
  EXPECT_FALSE(traversed({0, 0}, {0, 0}, {2, 0}, {-1, 0}));
}
